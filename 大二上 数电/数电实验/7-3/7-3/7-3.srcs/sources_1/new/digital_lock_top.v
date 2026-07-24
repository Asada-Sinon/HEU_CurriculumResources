`timescale 1ns / 1ps
// 顶层模块定义了系统的所有I/O接口，包括时钟和复位、用户输入（如按键和键盘）、显示输出（如数码管和LED）以及与蓝牙模块的通信接口
module digital_lock_top (
    input clk_100MHz,    
    input rst_n,           
    input [4:0] pb,                 // 按键
    input [2:0] sw_mode,            // 拨码开关
    //键盘
    input [3:0] row,
    output [3:0] col,
    // 数码管
    output CA0, CB0, CC0, CD0, CE0, CF0, CG0, DP0,
    output CA1, CB1, CC1, CD1, CE1, CF1, CG1, DP1,
    output BIT1, BIT2, BIT3, BIT4, BIT5, BIT6, BIT7, BIT8,
    output [15:0] led,
    // 蓝牙
    input [5:0] sw_pin,
    output bt_pw_on,
    output bt_master_slave,
    output bt_sw_hw,
    output bt_rst_n,
    output bt_sw,
    input ble_rx,
    output ble_tx
);
wire clk_2Hz;                       
wire [2:0] sw_mode_debounced; // 消抖后的拨码开关信号
wire password_correct; // 密码正确
wire alarm_signal; // 密码错误
wire [7:0] seg_dn0, seg_dn1;// 数码管段码
wire [7:0] dig_sel; // 数码管位选
wire system_locked; // 密码错了锁了
wire input_complete;// 输入完成了，之后开始该检验对不对了
wire [2:0] current_bit_pos;// 当前位位置
wire [2:0] current_state_debug;
wire [31:0] current_password;
wire [1:0] input_mode;// 输入模式：0=按键，1=键盘，2=蓝牙
// 键盘信号
wire key_flag;
wire [3:0] key_value;
// 蓝牙相关信号
wire warning;
wire ble_rx_done;
wire [7:0] ble_rx_data;
// 闪烁控制信号
wire [7:0] digit_blink_ctrl;// 8位数码管闪烁控制
wire [7:0] led_blink_ctrl;       // 8位LED闪烁控制
wire blink_clock;   // 闪烁时钟信号
wire attack_defense_mode;
wire [7:0] attack_count;
clock_divider u_clock_divider (
    .clk_in(clk_100MHz),
    .rst_n(rst_n),
    .clk_2Hz(clk_2Hz)
);
switch_debounce #(
    .DEBOUNCE_TIME(20_000_000)      // 200ms消抖时间 @ 100MHz
) u_switch_debounce (
    .clk(clk_100MHz),
    .rst_n(rst_n),
    .sw_in(sw_mode),                // 原始拨码开关信号
    .sw_out(sw_mode_debounced)      // 消抖后的拨码开关信号
);
keyboard_4x4 u_keyboard_4x4 (
    .clk(clk_100MHz),
    .reset_n(rst_n),
    .keyboard_row_x4_i(row),
    .keyboard_col_x4_o(col),
    .key_flag(key_flag),
    .key_value(key_value)
);
// 数字密码锁主控制模块 - 集成改进的按键检测
password_controller u_password_controller (
    .clk(clk_100MHz),
    .rst_n(rst_n),
    .pb(pb),                        // 直接连接原始按键信号，内部处理消抖
    .key_flag(key_flag),
    .key_value(key_value),
    .ble_rx_done(ble_rx_done),
    .ble_rx_data(ble_rx_data),
    .sw_mode(sw_mode_debounced),    // 使用消抖后的拨码开关信号
    .password_correct(password_correct),
    .alarm_signal(alarm_signal),
    .system_locked(system_locked),
    .input_complete(input_complete),
    .current_bit_pos(current_bit_pos),
    .current_state_debug(current_state_debug),
    .current_password(current_password),
    .input_mode(input_mode),
    .warning(warning),
    .blink_enable(),                // 内部使用，不连接
    .digit_blink_ctrl(digit_blink_ctrl), 
    .led_blink_ctrl(led_blink_ctrl),     
    .blink_clock(blink_clock),           
    .attack_defense_mode(attack_defense_mode), 
    .attack_count(attack_count)          
);
// 数码管显示控制模块
seg_display_controller u_seg_display (
    .clk(clk_100MHz),
    .clk_2Hz(clk_2Hz),
    .rst_n(rst_n),
    .password_correct(password_correct),
    .alarm_signal(alarm_signal),
    .system_locked(system_locked),
    .input_complete(input_complete),
    .current_bit_pos(current_bit_pos),
    .current_state_debug(current_state_debug),
    .current_password(current_password),
    .input_mode(input_mode),
    .digit_blink_ctrl(digit_blink_ctrl), 
    .blink_clock(blink_clock),           
    .seg_dn0(seg_dn0),
    .seg_dn1(seg_dn1),
    .dig_sel(dig_sel)
);
// LED控制模块
led_controller u_led_controller (
    .clk(clk_100MHz),
    .clk_2Hz(clk_2Hz),
    .rst_n(rst_n),
    .password_correct(password_correct),
    .alarm_signal(alarm_signal),
    .current_bit_pos(current_bit_pos),
    .current_state_debug(current_state_debug),
    .input_mode(input_mode),
    .led_blink_ctrl(led_blink_ctrl),     
    .blink_clock(blink_clock),           
    .led(led)
);
// 蓝牙串口发送模块
uart_data_tx u_uart_data_tx (
    .clk(clk_100MHz),
    .rst_n(rst_n),
    .tx_en(warning),
    .tx_num(8'd100),
    .tx_data({"warning!!!", 8'h0A, "warning!!!", 8'h0A, "warning!!!", 8'h0A, "!!!someone is trying to open your lock!!!"}),
    .Baud_Set(3'd0),
    .uart_tx_done(),
    .uart_tx(ble_tx)
);
// 蓝牙串口接收模块
uart_data_rx u_uart_data_rx (
    .clk(clk_100MHz),
    .rst_n(rst_n),
    .uart_rx(ble_rx),
    .Baud_Set(3'd0),
    .rx_data(ble_rx_data),
    .rx_done(ble_rx_done)
);
// 数码管
assign {CG0, CF0, CE0, CD0, CC0, CB0, CA0} = seg_dn0[6:0];
assign DP0 = 1'b0;  
assign {CG1, CF1, CE1, CD1, CC1, CB1, CA1} = seg_dn1[6:0];
assign DP1 = 1'b0;  
//位选
assign {BIT8, BIT7, BIT6, BIT5, BIT4, BIT3, BIT2, BIT1} = ~dig_sel;
assign bt_master_slave = sw_pin[0];
assign bt_sw_hw = sw_pin[1];
assign bt_rst_n = sw_pin[2];
assign bt_sw = sw_pin[3];
assign bt_pw_on = sw_pin[4];

endmodule