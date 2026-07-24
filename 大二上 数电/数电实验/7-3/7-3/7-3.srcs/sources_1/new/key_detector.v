`timescale 1ns / 1ps

// 按键检测模块 - 支持短按和长按检测
module key_detector(
    input clk,
    input rst_n,
    input key_in,
    output reg short_press,    // 短按脉冲
    output reg long_press      // 长按脉冲（重复）
);

parameter DEBOUNCE_TIME = 22'd4_999_999;    // 100ms@50MHz 消抖时间 (调整为100MHz: 9_999_999)
parameter LONG_PRESS_TIME = 26'd49_999_999; // 1秒长按判定 (调整为100MHz: 99_999_999)
parameter REPEAT_TIME = 24'd9_999_999;      // 100ms重复周期 (调整为100MHz: 19_999_999)

// 状态定义
parameter IDLE = 3'b000;
parameter DEBOUNCE_PRESS = 3'b001;
parameter PRESSED = 3'b010;
parameter LONG_ACTIVE = 3'b100;
parameter DEBOUNCE_RELEASE = 3'b101;

reg [2:0] state, next_state;
reg [26:0] counter;
reg key_sync1, key_sync2, key_sync3;
reg key_stable;

// 三级同步防抖
always @(posedge clk) begin
    if (!rst_n) begin
        key_sync1 <= 1'b1;
        key_sync2 <= 1'b1;
        key_sync3 <= 1'b1;
    end else begin
        key_sync1 <= key_in;
        key_sync2 <= key_sync1;
        key_sync3 <= key_sync2;
    end
end

// 稳定信号检测
always @(posedge clk) begin
    if (!rst_n) begin
        key_stable <= 1'b1;
    end else if (key_sync1 == key_sync2 && key_sync2 == key_sync3) begin
        key_stable <= key_sync3;
    end
end

// 统一的按键状态机 - 只在这里驱动输出信号，避免多驱动
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state <= IDLE;
        counter <= 27'd0;
        short_press <= 1'b0;
        long_press <= 1'b0;
    end else begin
        // 默认清除输出脉冲
        short_press <= 1'b0;
        long_press <= 1'b0;
        
        case (state)
            IDLE: begin
                counter <= 27'd0;
                if (!key_stable) begin  // 检测到按键按下（低电平有效）
                    state <= DEBOUNCE_PRESS;
                end
            end
            
            DEBOUNCE_PRESS: begin
                if (key_stable) begin  // 按键释放了，是误触
                    state <= IDLE;
                    counter <= 27'd0;
                end else if (counter >= DEBOUNCE_TIME) begin  // 消抖时间到，确认按下
                    state <= PRESSED;
                    counter <= 27'd0;
                end else begin
                    counter <= counter + 1'b1;
                end
            end
            
            PRESSED: begin
                if (key_stable) begin  // 按键释放，进入释放消抖
                    state <= DEBOUNCE_RELEASE;
                    counter <= 27'd0;
                end else if (counter >= LONG_PRESS_TIME) begin  // 长按时间到
                    state <= LONG_ACTIVE;
                    counter <= 27'd0;
                    long_press <= 1'b1;  // 第一次长按触发
                end else begin
                    counter <= counter + 1'b1;
                end
            end
            
            LONG_ACTIVE: begin
                if (key_stable) begin  // 按键释放
                    state <= DEBOUNCE_RELEASE;
                    counter <= 27'd0;
                end else if (counter >= REPEAT_TIME) begin  // 重复触发
                    counter <= 27'd0;
                    long_press <= 1'b1;  // 连续触发长按
                end else begin
                    counter <= counter + 1'b1;
                end
            end
            
            DEBOUNCE_RELEASE: begin
                if (!key_stable) begin  // 又按下了，回到按下状态
                    state <= PRESSED;
                    counter <= 27'd0;
                end else if (counter >= DEBOUNCE_TIME) begin  // 确认释放
                    state <= IDLE;
                    counter <= 27'd0;
                    // 只有从PRESSED状态释放的才产生短按脉冲
                    short_press <= 1'b1;  // 产生短按脉冲
                end else begin
                    counter <= counter + 1'b1;
                end
            end
            
            default: begin
                state <= IDLE;
                counter <= 27'd0;
            end
        endcase
    end
end

endmodule