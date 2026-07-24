// 数码管显示控制模块 - 修复闪烁逻辑
module seg_display_controller (
    input clk,
    input clk_2Hz,
    input rst_n,
    input password_correct,
    input alarm_signal,
    input system_locked,
    input input_complete,
    input [2:0] current_bit_pos,
    input [2:0] current_state_debug,
    input [31:0] current_password,    // 32位十进制密码
    input [1:0] input_mode,           // 修改为2位：0=按键，1=键盘，2=蓝牙
    input [7:0] digit_blink_ctrl,     // 新增：8位数码管闪烁控制
    input blink_clock,                // 新增：闪烁时钟信号
    output reg [7:0] seg_dn0,
    output reg [7:0] seg_dn1,
    output reg [7:0] dig_sel
);

// 扫描计数器和显示位置
reg [2:0] scan_count;
reg [2:0] move_position;        // 移动字符位置

// 提取各位十进制数字
wire [3:0] digit_7 = current_password[31:28];  // 位7, 显示在BIT1
wire [3:0] digit_6 = current_password[27:24];  // 位6, 显示在BIT2
wire [3:0] digit_5 = current_password[23:20];  // 位5, 显示在BIT3
wire [3:0] digit_4 = current_password[19:16];  // 位4, 显示在BIT4
wire [3:0] digit_3 = current_password[15:12];  // 位3, 显示在BIT5
wire [3:0] digit_2 = current_password[11:8];   // 位2, 显示在BIT6
wire [3:0] digit_1 = current_password[7:4];    // 位1, 显示在BIT7
wire [3:0] digit_0 = current_password[3:0];    // 位0, 显示在BIT8

// 2kHz扫描时钟生成
reg [15:0] scan_counter;
reg scan_clk;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        scan_counter <= 16'd0;
        scan_clk <= 1'b0;
    end else begin
        if (scan_counter >= 16'd49_999) begin // 100MHz/50000 = 2kHz
            scan_counter <= 16'd0;
            scan_clk <= ~scan_clk;
        end else begin
            scan_counter <= scan_counter + 1'b1;
        end
    end
end

// 扫描计数器
always @(posedge scan_clk or negedge rst_n) begin
    if (!rst_n) begin
        scan_count <= 3'd0;
    end else begin
        if (scan_count >= 3'd7) begin
            scan_count <= 3'd0;
        end else begin
            scan_count <= scan_count + 1'b1;
        end
    end
end

// 移动字符控制 - 使用2Hz时钟
always @(posedge clk_2Hz or negedge rst_n) begin
    if (!rst_n) begin
        move_position <= 3'd0;
    end else begin
        if (move_position >= 3'd7) begin
            move_position <= 3'd0;
        end else begin
            move_position <= move_position + 1'b1;
        end
    end
end

// 段码解码函数
function [7:0] digit_to_segments;
    input [3:0] digit;
    begin
        case (digit)
            4'h0: digit_to_segments = 8'b00111111; // 0
            4'h1: digit_to_segments = 8'b00000110; // 1
            4'h2: digit_to_segments = 8'b01011011; // 2
            4'h3: digit_to_segments = 8'b01001111; // 3
            4'h4: digit_to_segments = 8'b01100110; // 4
            4'h5: digit_to_segments = 8'b01101101; // 5
            4'h6: digit_to_segments = 8'b01111101; // 6
            4'h7: digit_to_segments = 8'b00000111; // 7
            4'h8: digit_to_segments = 8'b01111111; // 8
            4'h9: digit_to_segments = 8'b01101111; // 9
            default: digit_to_segments = 8'b00000000; // 关闭
        endcase
    end
endfunction

// 字符段码解码函数
function [7:0] char_to_segments;
    input [4:0] char_code;
    begin
        case (char_code)
            // 数字0-9
            5'h00: char_to_segments = 8'b00111111; // 0
            5'h01: char_to_segments = 8'b00000110; // 1
            5'h02: char_to_segments = 8'b01011011; // 2
            5'h03: char_to_segments = 8'b01001111; // 3
            5'h04: char_to_segments = 8'b01100110; // 4
            5'h05: char_to_segments = 8'b01101101; // 5
            5'h06: char_to_segments = 8'b01111101; // 6
            5'h07: char_to_segments = 8'b00000111; // 7
            5'h08: char_to_segments = 8'b01111111; // 8
            5'h09: char_to_segments = 8'b01101111; // 9
            // 字母
            5'h10: char_to_segments = 8'b00111111; // O
            5'h11: char_to_segments = 8'b01110011; // P
            5'h12: char_to_segments = 8'b01111001; // E
            5'h13: char_to_segments = 8'b01010100; // n
            5'h14: char_to_segments = 8'b01010000; // r
            5'h15: char_to_segments = 8'b01011100; // o
            5'h16: char_to_segments = 8'b00111000; // L
            5'h17: char_to_segments = 8'b00111001; // C
            5'h18: char_to_segments = 8'b01110110; // H
            default: char_to_segments = 8'b00000000; // 关闭
        endcase
    end
endfunction

// 主要的显示控制逻辑
always @(*) begin
    // 默认位选控制
    case (scan_count)
        3'd0: dig_sel = 8'b11111110; // BIT1 - 显示位7
        3'd1: dig_sel = 8'b11111101; // BIT2 - 显示位6
        3'd2: dig_sel = 8'b11111011; // BIT3 - 显示位5
        3'd3: dig_sel = 8'b11110111; // BIT4 - 显示位4
        3'd4: dig_sel = 8'b11101111; // BIT5 - 显示位3
        3'd5: dig_sel = 8'b11011111; // BIT6 - 显示位2
        3'd6: dig_sel = 8'b10111111; // BIT7 - 显示位1
        3'd7: dig_sel = 8'b01111111; // BIT8 - 显示位0
    endcase
    
    // 默认段码
    seg_dn0 = 8'b00000000;
    seg_dn1 = 8'b00000000;
    
    // 根据状态决定显示内容
    if (password_correct && !system_locked) begin
        // 显示"OPEN"滚动移动
        case ((scan_count + move_position) % 8)
            3'd0: begin seg_dn0 = char_to_segments(5'h10); seg_dn1 = char_to_segments(5'h10); end // O
            3'd1: begin seg_dn0 = char_to_segments(5'h11); seg_dn1 = char_to_segments(5'h11); end // P
            3'd2: begin seg_dn0 = char_to_segments(5'h12); seg_dn1 = char_to_segments(5'h12); end // E
            3'd3: begin seg_dn0 = char_to_segments(5'h13); seg_dn1 = char_to_segments(5'h13); end // n
            default: begin seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000; end // 关闭
        endcase
    end else if (alarm_signal) begin
        // 显示"Error"滚动移动
        case ((scan_count + move_position) % 8)
            3'd0: begin seg_dn0 = char_to_segments(5'h12); seg_dn1 = char_to_segments(5'h12); end // E
            3'd1: begin seg_dn0 = char_to_segments(5'h14); seg_dn1 = char_to_segments(5'h14); end // r
            3'd2: begin seg_dn0 = char_to_segments(5'h14); seg_dn1 = char_to_segments(5'h14); end // r
            3'd3: begin seg_dn0 = char_to_segments(5'h15); seg_dn1 = char_to_segments(5'h15); end // o
            3'd4: begin seg_dn0 = char_to_segments(5'h14); seg_dn1 = char_to_segments(5'h14); end // r
            default: begin seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000; end // 关闭
        endcase
    end else if (current_state_debug == 3'b001 || current_state_debug == 3'b010) begin
        // 设置模式或输入模式：显示8位十进制密码
        case (scan_count)
            3'd0: begin // BIT1显示位7
                seg_dn0 = digit_to_segments(digit_7);
                seg_dn1 = digit_to_segments(digit_7);
                // 检查是否需要闪烁 (位7对应digit_blink_ctrl[7])
                if (digit_blink_ctrl[7] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd1: begin // BIT2显示位6
                seg_dn0 = digit_to_segments(digit_6);
                seg_dn1 = digit_to_segments(digit_6);
                // 检查是否需要闪烁 (位6对应digit_blink_ctrl[6])
                if (digit_blink_ctrl[6] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd2: begin // BIT3显示位5
                seg_dn0 = digit_to_segments(digit_5);
                seg_dn1 = digit_to_segments(digit_5);
                // 检查是否需要闪烁 (位5对应digit_blink_ctrl[5])
                if (digit_blink_ctrl[5] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd3: begin // BIT4显示位4
                seg_dn0 = digit_to_segments(digit_4);
                seg_dn1 = digit_to_segments(digit_4);
                // 检查是否需要闪烁 (位4对应digit_blink_ctrl[4])
                if (digit_blink_ctrl[4] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd4: begin // BIT5显示位3
                seg_dn0 = digit_to_segments(digit_3);
                seg_dn1 = digit_to_segments(digit_3);
                // 检查是否需要闪烁 (位3对应digit_blink_ctrl[3])
                if (digit_blink_ctrl[3] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd5: begin // BIT6显示位2
                seg_dn0 = digit_to_segments(digit_2);
                seg_dn1 = digit_to_segments(digit_2);
                // 检查是否需要闪烁 (位2对应digit_blink_ctrl[2])
                if (digit_blink_ctrl[2] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd6: begin // BIT7显示位1
                seg_dn0 = digit_to_segments(digit_1);
                seg_dn1 = digit_to_segments(digit_1);
                // 检查是否需要闪烁 (位1对应digit_blink_ctrl[1])
                if (digit_blink_ctrl[1] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
            3'd7: begin // BIT8显示位0
                seg_dn0 = digit_to_segments(digit_0);
                seg_dn1 = digit_to_segments(digit_0);
                // 检查是否需要闪烁 (位0对应digit_blink_ctrl[0])
                if (digit_blink_ctrl[0] && !blink_clock) begin
                    seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000;
                end
            end
        endcase
    end else begin
        // 空闲状态：显示"LOCH"滚动移动
        case ((scan_count + move_position) % 8)
            3'd0: begin seg_dn0 = char_to_segments(5'h16); seg_dn1 = char_to_segments(5'h16); end // L
            3'd1: begin seg_dn0 = char_to_segments(5'h10); seg_dn1 = char_to_segments(5'h10); end // O
            3'd2: begin seg_dn0 = char_to_segments(5'h17); seg_dn1 = char_to_segments(5'h17); end // C
            3'd3: begin seg_dn0 = char_to_segments(5'h18); seg_dn1 = char_to_segments(5'h18); end // H
            default: begin seg_dn0 = 8'b00000000; seg_dn1 = 8'b00000000; end // 关闭
        endcase
    end
end

endmodule