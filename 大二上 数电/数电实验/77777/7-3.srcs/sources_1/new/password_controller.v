`timescale 1ns / 1ps

//==============================================================================
// 密码控制器主模块 - 完整修复版本
//==============================================================================
module password_controller (
    input clk,
    input rst_n,
    input [4:0] pb,                 // 5个按键原始输入
    input key_flag,                 // 4×4键盘按键标志
    input [3:0] key_value,          // 4×4键盘按键值
    input ble_rx_done,              // 蓝牙接收完成
    input [7:0] ble_rx_data,        // 蓝牙接收数据
    input [2:0] sw_mode,            // 3位拨码开关控制模式（已消抖）
    output reg password_correct,    // 密码正确输出
    output reg alarm_signal,        // 报警信号输出
    output reg system_locked,       // 系统锁定状态
    output reg input_complete,      // 输入完成信号
    output reg [2:0] current_bit_pos, // 当前位位置输出
    output reg [2:0] current_state_debug, // 当前状态输出
    output reg [31:0] current_password, // 当前正在编辑的密码
    output reg [1:0] input_mode,    // 输入模式
    output reg warning,             // 警告信号
    output reg blink_enable,        // 闪烁使能信号
    output reg [7:0] digit_blink_ctrl, // 数码管闪烁控制
    output reg [7:0] led_blink_ctrl,   // LED闪烁控制
    output wire blink_clock,        // 闪烁时钟输出
    output reg attack_defense_mode, // 攻防模式标志
    output reg [7:0] attack_count   // 攻击计数器
);

// 按键功能重新定义 - 参考时钟系统
wire right_key = ~pb[0];        // PB0: 向右移位 
wire dec_key = ~pb[1];          // PB1: 递减数值
wire enter_key = ~pb[2];        // PB2: 确认当前输入
wire left_key = ~pb[3];         // PB3: 向左移位
wire inc_key = ~pb[4];          // PB4: 递增数值

// 按键检测信号
wire right_short, right_long;
wire dec_short, dec_long;
wire enter_short, enter_long;
wire left_short, left_long;
wire inc_short, inc_long;

// 按键触发信号（合并短按和长按）
wire right_trigger = right_short || right_long;
wire dec_trigger = dec_short || dec_long;
wire enter_trigger = enter_short;  // 确认键只响应短按
wire left_trigger = left_short || left_long;
wire inc_trigger = inc_short || inc_long;

// 内部寄存器声明
reg [31:0] stored_password;     
reg [31:0] input_password;      
reg [31:0] temp_password;       
reg [2:0] input_bit_count;      
reg [31:0] ble_password;        
reg [3:0] ble_byte_cnt;         
reg ble_set_mode;               
reg [3:0] keypad_input_count;   
reg keyboard_detected;          
reg mode_switch_detected;       
reg [3:0] reset_stable_count;
reg [25:0] blink_counter;       
reg blink_state;                
reg [15:0] attack_timer;        
reg [3:0] failed_attempts;      
reg [31:0] brute_force_password; // 储存当前正在尝试的密码
reg [24:0] speed_counter;        // 用于控制暴力破解速度的计数器
wire brute_force_tick;           // 速度控制信号，为高时表示可以更新密码

parameter MODE_BRUTE_FORCE = 2'b11; // 暴力破解专用输入模式标识
// 设置模式控制 - 借鉴时钟系统的设置逻辑
reg time_setting_active;        // 时间设置激活状态
reg setting_confirmed;          // 设置确认标志

wire system_stable = (reset_stable_count >= 4'd15);

// 状态机定义
parameter IDLE = 3'b000;        
parameter SET_PASSWORD = 3'b001; 
parameter INPUT_PASSWORD = 3'b010; 
parameter CHECK_PASSWORD = 3'b011; 
parameter UNLOCK = 3'b100;      
parameter ALARM = 3'b101;       
parameter ATTACK_DEFENSE = 3'b110; 

reg [2:0] current_state, next_state;

// 按键检测模块实例化
key_detector u_right_key(
    .clk(clk),
    .rst_n(rst_n),
    .key_in(right_key),
    .short_press(right_short),
    .long_press(right_long)
);

key_detector u_dec_key(
    .clk(clk),
    .rst_n(rst_n),
    .key_in(dec_key),
    .short_press(dec_short),
    .long_press(dec_long)
);

key_detector u_enter_key(
    .clk(clk),
    .rst_n(rst_n),
    .key_in(enter_key),
    .short_press(enter_short),
    .long_press(enter_long)
);

key_detector u_left_key(
    .clk(clk),
    .rst_n(rst_n),
    .key_in(left_key),
    .short_press(left_short),
    .long_press(left_long)
);

key_detector u_inc_key(
    .clk(clk),
    .rst_n(rst_n),
    .key_in(inc_key),
    .short_press(inc_short),
    .long_press(inc_long)
);

// 键盘映射函数
function [3:0] keypad_to_decimal;
    input [3:0] key_val;
    begin
        case(key_val)
            4'h1: keypad_to_decimal = 4'h1;
            4'h2: keypad_to_decimal = 4'h4;
            4'h3: keypad_to_decimal = 4'h7;
            4'h4: keypad_to_decimal = 4'h2;
            4'h5: keypad_to_decimal = 4'h5;
            4'h6: keypad_to_decimal = 4'h8;
            4'h7: keypad_to_decimal = 4'h3;
            4'h8: keypad_to_decimal = 4'h6;
            4'h9: keypad_to_decimal = 4'h9;
            4'hb: keypad_to_decimal = 4'h0;
            default: keypad_to_decimal = 4'hf;
        endcase
    end
endfunction

// BCD码递增函数
function [31:0] bcd_increment(input [31:0] current_val);
    integer i;
    reg [3:0] d [0:7];
    reg [7:0] carry;        // 8-bit vector 用作每位进位标志
    reg [31:0] tmp;
begin
    // 把每个 nibble 提取到数组 d 中（低位在 d[0]）
    tmp = current_val;
    for (i = 0; i < 8; i = i + 1) begin
        d[i] = tmp[3:0];
        tmp = tmp >> 4;
        carry[i] = 1'b0;
    end

    // 最低位 +1，处理进位
    d[0] = d[0] + 1'b1;
    carry[0] = (d[0] > 4'd9);
    if (carry[0]) d[0] = 4'd0;

    // 其余位按进位传播
    for (i = 1; i < 8; i = i + 1) begin
        d[i] = d[i] + carry[i-1];
        carry[i] = (d[i] > 4'd9);
        if (carry[i]) d[i] = 4'd0;
    end

    // 组合回 32 位 BCD 值（d[7] 为最高位）
    bcd_increment = {d[7], d[6], d[5], d[4], d[3], d[2], d[1], d[0]};
end
endfunction

// 破解速度控制器
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        speed_counter <= 25'd0;
    end else if (current_state == INPUT_PASSWORD && sw_mode[1]) begin
        if (speed_counter >= 25'd2500000) // 调整此值以改变速度
            speed_counter <= 25'd0;
        else
            speed_counter <= speed_counter + 1'b1;
    end else begin
        speed_counter <= 25'd0;
    end
end
assign brute_force_tick = (speed_counter == 25'd2500000);

// 闪烁时钟生成 (2Hz)
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        blink_counter <= 26'b0;
        blink_state <= 1'b0;
    end else begin
        if (blink_counter >= 26'd25000000) begin
            blink_counter <= 26'b0;
            blink_state <= ~blink_state;
        end else begin
            blink_counter <= blink_counter + 1'b1;
        end
    end
end

assign blink_clock = blink_state;

// 闪烁控制逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        digit_blink_ctrl <= 8'b00000000;
        led_blink_ctrl <= 8'b00000000;
    end else begin
        digit_blink_ctrl <= 8'b00000000;
        led_blink_ctrl <= 8'b00000000;
        
        if ((current_state == SET_PASSWORD && time_setting_active) || 
            (current_state == INPUT_PASSWORD && time_setting_active)) begin
            
            case (input_bit_count)
                3'b000: begin 
                    digit_blink_ctrl[7] <= 1'b1;
                    led_blink_ctrl[0] <= 1'b1;
                end
                3'b001: begin 
                    digit_blink_ctrl[6] <= 1'b1;
                    led_blink_ctrl[1] <= 1'b1;
                end
                3'b010: begin 
                    digit_blink_ctrl[5] <= 1'b1;
                    led_blink_ctrl[2] <= 1'b1;
                end
                3'b011: begin 
                    digit_blink_ctrl[4] <= 1'b1;
                    led_blink_ctrl[3] <= 1'b1;
                end
                3'b100: begin 
                    digit_blink_ctrl[3] <= 1'b1;
                    led_blink_ctrl[4] <= 1'b1;
                end
                3'b101: begin 
                    digit_blink_ctrl[2] <= 1'b1;
                    led_blink_ctrl[5] <= 1'b1;
                end
                3'b110: begin 
                    digit_blink_ctrl[1] <= 1'b1;
                    led_blink_ctrl[6] <= 1'b1;
                end
                3'b111: begin 
                    digit_blink_ctrl[0] <= 1'b1;
                    led_blink_ctrl[7] <= 1'b1;
                end
            endcase
        end else if (current_state == ATTACK_DEFENSE) begin
            digit_blink_ctrl <= 8'b11111111;
            led_blink_ctrl <= 8'b11111111;
        end
    end
end

// 复位稳定计数器
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        reset_stable_count <= 4'b0;
    end else if (!system_stable) begin
        reset_stable_count <= reset_stable_count + 1'b1;
    end
end

// 蓝牙数据接收处理
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        ble_password <= 32'h0;
        ble_byte_cnt <= 4'd0;
        ble_set_mode <= 1'b0;
    end
    else if(ble_rx_done && system_stable) begin
        // 检测蓝牙设置命令 "SET:"
        if(ble_byte_cnt == 4'd0 && ble_rx_data == 8'h53) begin // 'S'
            ble_set_mode <= 1'b1;
            ble_byte_cnt <= ble_byte_cnt + 1'b1;
        end
        else if(ble_byte_cnt == 4'd1 && ble_rx_data == 8'h45 && ble_set_mode) begin // 'E'
            ble_byte_cnt <= ble_byte_cnt + 1'b1;
        end
        else if(ble_byte_cnt == 4'd2 && ble_rx_data == 8'h54 && ble_set_mode) begin // 'T'
            ble_byte_cnt <= ble_byte_cnt + 1'b1;
        end
        else if(ble_byte_cnt == 4'd3 && ble_rx_data == 8'h3A && ble_set_mode) begin // ':'
            ble_byte_cnt <= ble_byte_cnt + 1'b1;
        end
        else if(ble_byte_cnt >= 4'd4 && ble_byte_cnt < 4'd12 && ble_set_mode) begin
            case(ble_byte_cnt - 4'd4)
                4'd0: ble_password[31:28] <= ble_rx_data[3:0];
                4'd1: ble_password[27:24] <= ble_rx_data[3:0];
                4'd2: ble_password[23:20] <= ble_rx_data[3:0];
                4'd3: ble_password[19:16] <= ble_rx_data[3:0];
                4'd4: ble_password[15:12] <= ble_rx_data[3:0];
                4'd5: ble_password[11:8] <= ble_rx_data[3:0];
                4'd6: ble_password[7:4] <= ble_rx_data[3:0];
                4'd7: ble_password[3:0] <= ble_rx_data[3:0];
            endcase
            
            if(ble_byte_cnt == 4'd11) begin
                ble_byte_cnt <= 4'd0;
                ble_set_mode <= 1'b0;
            end else begin
                ble_byte_cnt <= ble_byte_cnt + 1'b1;
            end
        end
        else if(!ble_set_mode) begin
            case(ble_byte_cnt)
                4'd0: ble_password[31:28] <= ble_rx_data[3:0];
                4'd1: ble_password[27:24] <= ble_rx_data[3:0];
                4'd2: ble_password[23:20] <= ble_rx_data[3:0];
                4'd3: ble_password[19:16] <= ble_rx_data[3:0];
                4'd4: ble_password[15:12] <= ble_rx_data[3:0];
                4'd5: ble_password[11:8] <= ble_rx_data[3:0];
                4'd6: ble_password[7:4] <= ble_rx_data[3:0];
                4'd7: ble_password[3:0] <= ble_rx_data[3:0];
            endcase
            
            if(ble_byte_cnt >= 4'd7)
                ble_byte_cnt <= 4'd0;
            else
                ble_byte_cnt <= ble_byte_cnt + 1'b1;
        end
        else begin
            ble_byte_cnt <= 4'd0;
            ble_set_mode <= 1'b0;
        end
    end
end

// 统一的状态机时序逻辑 - 完整修复版本
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state <= IDLE;
        stored_password <= 32'h01234567;
        input_password <= 32'h00000000;
        temp_password <= 32'h00000000;
        input_bit_count <= 3'b000;
        password_correct <= 1'b0;
        alarm_signal <= 1'b0;
        system_locked <= 1'b1;
        input_complete <= 1'b0;
        current_bit_pos <= 3'b000;
        current_password <= 32'haaaaaaaa;
        input_mode <= 2'b00;
        warning <= 1'b0;
        keypad_input_count <= 4'b0;
        blink_enable <= 1'b0;
        attack_defense_mode <= 1'b0;
        attack_count <= 8'b0;
        attack_timer <= 16'b0;
        failed_attempts <= 4'b0;
        keyboard_detected <= 1'b0;
        mode_switch_detected <= 1'b0;
        time_setting_active <= 1'b0;
        setting_confirmed <= 1'b0;
        brute_force_password <= 32'h00000000;
    end else begin
        if (system_stable) begin
            // 输入模式检测逻辑
            if(sw_mode[0] || sw_mode[1] || sw_mode[2]) begin
                mode_switch_detected <= 1'b1;
                keyboard_detected <= 1'b0;
            end else if(sw_mode == 3'b000) begin
                mode_switch_detected <= 1'b0;
                if(key_flag && keypad_to_decimal(key_value) < 4'hf && current_state == IDLE) begin
                    keyboard_detected <= 1'b1;
                end else if(current_state != IDLE) begin
                    keyboard_detected <= 1'b0;
                end
            end
            
            // 状态转换
            current_state <= next_state;
        end
        
        // 更新状态显示
        current_state_debug <= current_state;
        
        case (current_state)
            IDLE: begin
                brute_force_password <= 32'h00000000;
                alarm_signal <= 1'b0;
                input_complete <= 1'b0;
                current_bit_pos <= 3'b000;
                current_password <= 32'haaaaaaaa;
                input_mode <= 2'b00;
                warning <= 1'b0;
                blink_enable <= 1'b0;
                attack_defense_mode <= sw_mode[2];
                
                if (sw_mode == 3'b000) begin
                    input_password <= 32'h00000000;
                    temp_password <= 32'h00000000;
                    input_bit_count <= 3'b000;
                    keypad_input_count <= 4'b0;
                    time_setting_active <= 1'b0;
                    setting_confirmed <= 1'b0;
                end
            end
            
            SET_PASSWORD: begin
                blink_enable <= time_setting_active;
                attack_defense_mode <= 1'b0;
                
                // 密码保存逻辑
                if (setting_confirmed) begin
                    stored_password <= temp_password;  // 保存临时密码到存储密码
                    setting_confirmed <= 1'b0;         // 清除确认标志
                end
                
                if (ble_set_mode && ble_byte_cnt == 4'd0 && ble_rx_done) begin
                    stored_password <= ble_password;
                    input_complete <= 1'b1;
                    input_mode <= 2'b10;
                    time_setting_active <= 1'b0;
                end else begin
                    input_mode <= 2'b00;
                    current_bit_pos <= input_bit_count;
                    current_password <= temp_password;
                    
                    // 进入设置模式时初始化临时密码并自动激活
                    if (temp_password == 32'h00000000 && input_bit_count == 3'b000 && !time_setting_active) begin
                        temp_password <= stored_password;
                        time_setting_active <= 1'b1;  // 自动激活设置模式
                        input_bit_count <= 3'b000;    // 从第一位开始
                    end
                    
                    // PB2确认设置完成
                    if (enter_trigger) begin
                        if (time_setting_active) begin
                            // 确认设置完成
                            time_setting_active <= 1'b0;
                            setting_confirmed <= 1'b1;   // 设置确认标志
                            input_complete <= 1'b1;      // 设置完成标志
                        end
                    end
                    
                    // 位选移动和数值调整（只在激活状态下有效）
                    if (time_setting_active) begin
                        if (left_trigger) begin  // PB3: 左移
                            input_bit_count <= (input_bit_count == 3'b000) ? 3'b111 : input_bit_count - 1'b1;
                        end
                        
                        if (right_trigger) begin  // PB0: 右移
                            input_bit_count <= (input_bit_count == 3'b111) ? 3'b000 : input_bit_count + 1'b1;
                        end
                        
                        // 数值调整 - PB4递增，PB1递减
                        if (inc_trigger) begin
                            case (input_bit_count)
                                3'b000: temp_password[31:28] <= (temp_password[31:28] == 4'd9) ? 4'd0 : temp_password[31:28] + 1'b1;
                                3'b001: temp_password[27:24] <= (temp_password[27:24] == 4'd9) ? 4'd0 : temp_password[27:24] + 1'b1;
                                3'b010: temp_password[23:20] <= (temp_password[23:20] == 4'd9) ? 4'd0 : temp_password[23:20] + 1'b1;
                                3'b011: temp_password[19:16] <= (temp_password[19:16] == 4'd9) ? 4'd0 : temp_password[19:16] + 1'b1;
                                3'b100: temp_password[15:12] <= (temp_password[15:12] == 4'd9) ? 4'd0 : temp_password[15:12] + 1'b1;
                                3'b101: temp_password[11:8] <= (temp_password[11:8] == 4'd9) ? 4'd0 : temp_password[11:8] + 1'b1;
                                3'b110: temp_password[7:4] <= (temp_password[7:4] == 4'd9) ? 4'd0 : temp_password[7:4] + 1'b1;
                                3'b111: temp_password[3:0] <= (temp_password[3:0] == 4'd9) ? 4'd0 : temp_password[3:0] + 1'b1;
                            endcase
                        end
                        
                        if (dec_trigger) begin
                            case (input_bit_count)
                                3'b000: temp_password[31:28] <= (temp_password[31:28] == 4'd0) ? 4'd9 : temp_password[31:28] - 1'b1;
                                3'b001: temp_password[27:24] <= (temp_password[27:24] == 4'd0) ? 4'd9 : temp_password[27:24] - 1'b1;
                                3'b010: temp_password[23:20] <= (temp_password[23:20] == 4'd0) ? 4'd9 : temp_password[23:20] - 1'b1;
                                3'b011: temp_password[19:16] <= (temp_password[19:16] == 4'd0) ? 4'd9 : temp_password[19:16] - 1'b1;
                                3'b100: temp_password[15:12] <= (temp_password[15:12] == 4'd0) ? 4'd9 : temp_password[15:12] - 1'b1;
                                3'b101: temp_password[11:8] <= (temp_password[11:8] == 4'd0) ? 4'd9 : temp_password[11:8] - 1'b1;
                                3'b110: temp_password[7:4] <= (temp_password[7:4] == 4'd0) ? 4'd9 : temp_password[7:4] - 1'b1;
                                3'b111: temp_password[3:0] <= (temp_password[3:0] == 4'd0) ? 4'd9 : temp_password[3:0] - 1'b1;
                            endcase
                        end
                    end
                end
            end
            
            INPUT_PASSWORD: begin
                input_mode <= MODE_BRUTE_FORCE;           // 设置为暴力破解模式
                attack_defense_mode <= 1'b0;
                blink_enable <= 1'b0;                     // 破解时不需要闪烁
                current_password <= brute_force_password; // 在显示器上显示当前尝试的密码
                
                // 当速度控制器触发时 (brute_force_tick为高)
                if (brute_force_tick) begin
                    // 检查当前尝试的密码是否正确
                    if (brute_force_password == stored_password) begin
                        input_complete <= 1'b1;               // 破解成功，设置完成标志以跳转到UNLOCK状态
                        input_password <= brute_force_password; // 将找到的密码载入
                    end 
                    // 如果密码错误且未达到最大值(99999999)，则计算下一个密码
                    else if (brute_force_password != 32'h99999999) begin
                        brute_force_password <= bcd_increment(brute_force_password);
                    end
                end
            end
            
            CHECK_PASSWORD: begin
                blink_enable <= 1'b0;
                attack_defense_mode <= sw_mode[2];
                
                if (input_password == stored_password) begin
                    password_correct <= 1'b1;
                    system_locked <= 1'b0;
                    failed_attempts <= 4'b0;
                end else begin
                    alarm_signal <= 1'b1;
                    system_locked <= 1'b1;
                    warning <= 1'b1;
                    failed_attempts <= failed_attempts + 1'b1;
                end
                input_complete <= 1'b0;
                current_bit_pos <= 3'b000;
            end
            
            UNLOCK: begin
                blink_enable <= 1'b0;
                password_correct <= 1'b1;
                system_locked <= 1'b0;
                alarm_signal <= 1'b0;
                current_bit_pos <= 3'b000;
                current_password <= stored_password;
                warning <= 1'b0;
                attack_defense_mode <= sw_mode[2];
                attack_count <= failed_attempts;
                
                // 添加PB2按键响应 - 退回到闲置状态
                if (enter_trigger) begin
                    password_correct <= 1'b0;        // 清除密码正确标志
                    system_locked <= 1'b1;          // 重新锁定系统
                    current_password <= 32'haaaaaaaa; // 重置显示密码
                end
            end
            
            ALARM: begin
                blink_enable <= 1'b0;
                password_correct <= 1'b0;
                system_locked <= 1'b1;
                alarm_signal <= 1'b1;
                current_bit_pos <= 3'b000;
                current_password <= 32'h00000000;
                attack_defense_mode <= sw_mode[2];
                attack_count <= failed_attempts;
            end
            
            ATTACK_DEFENSE: begin
                blink_enable <= 1'b1;
                attack_defense_mode <= 1'b1;
                current_password <= 32'hdeadbeef;
                attack_count <= failed_attempts;
                
                if (attack_timer < 16'hFFFF) begin
                    attack_timer <= attack_timer + 1'b1;
                end
                
                if (enter_trigger) begin
                    failed_attempts <= 4'b0;
                    attack_count <= 8'b0;
                end
                
                if (inc_trigger) begin
                    if (failed_attempts < 4'hF) begin
                        failed_attempts <= failed_attempts + 1'b1;
                        attack_count <= attack_count + 1'b1;
                    end
                end
            end
        endcase
    end
end

// 状态机组合逻辑
always @(*) begin
    next_state = current_state;
    
    if (system_stable) begin
        case (current_state)
            IDLE: begin
                if (sw_mode == 3'b000) begin
                    if (keyboard_detected) begin
                        next_state = INPUT_PASSWORD;
                    end else if (ble_byte_cnt == 4'd0 && ble_rx_done && !ble_set_mode) begin
                        next_state = INPUT_PASSWORD;
                    end else if (ble_set_mode && ble_byte_cnt == 4'd0 && ble_rx_done) begin
                        next_state = SET_PASSWORD;
                    end else begin
                        next_state = IDLE;
                    end
                end else begin
                    if (sw_mode[2]) begin
                        next_state = ATTACK_DEFENSE;
                    end else if (sw_mode[0]) begin
                        next_state = SET_PASSWORD;
                    end else if (sw_mode[1]) begin
                        next_state = INPUT_PASSWORD;
                    end
                end
            end
            
            SET_PASSWORD: begin
                if (!sw_mode[0] && !ble_set_mode) begin
                    next_state = IDLE;
                end else if (input_complete) begin
                    next_state = IDLE;
                end
            end
            
            INPUT_PASSWORD: begin
                if (!sw_mode[1] && input_mode != 2'b01 && input_mode != 2'b10) begin
                    next_state = IDLE;
                end else if (input_complete) begin
                    next_state = CHECK_PASSWORD;
                end
            end
            
            CHECK_PASSWORD: begin
                if (input_password == stored_password) begin
                    next_state = UNLOCK;
                end else begin
                    next_state = ALARM;
                end
            end
            
            UNLOCK: begin
                if (sw_mode[2]) begin
                    next_state = ATTACK_DEFENSE;
                end else if (sw_mode[0]) begin
                    next_state = SET_PASSWORD;
                end else if (sw_mode[1]) begin
                    next_state = INPUT_PASSWORD;
                end else if (enter_trigger) begin    // 添加PB2按键响应
                    next_state = IDLE;
                end else begin
                    next_state = UNLOCK;             // 保持当前状态
                end
            end
            
            ALARM: begin
                if (sw_mode[2]) begin
                    next_state = ATTACK_DEFENSE;
                end else if (sw_mode[0]) begin
                    next_state = SET_PASSWORD;
                end else if (sw_mode[1]) begin
                    next_state = INPUT_PASSWORD;
                end else begin
                    if (enter_trigger) begin
                        next_state = IDLE;
                    end else begin
                        next_state = IDLE;
                    end
                end
            end
            
            ATTACK_DEFENSE: begin
                if (!sw_mode[2]) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
end

endmodule