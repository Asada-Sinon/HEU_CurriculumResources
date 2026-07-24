`timescale 1ns / 1ps

// 拨码开关消抖模块
module switch_debounce #(
    parameter DEBOUNCE_TIME = 20_000_000  // 消抖时间：200ms @ 100MHz
)(
    input clk,              // 时钟信号
    input rst_n,            // 复位信号（低有效）
    input [2:0] sw_in,      // 原始拨码开关输入
    output reg [2:0] sw_out // 消抖后的拨码开关输出
);

// 每个拨码开关的消抖计数器和状态
reg [24:0] debounce_counter [2:0];  // 3个拨码开关的消抖计数器
reg [2:0] sw_sync [1:0];            // 两级同步寄存器
reg [2:0] sw_stable;                // 稳定的开关状态

// 参数定义
localparam COUNTER_MAX = DEBOUNCE_TIME - 1;

integer i;

// 同步输入信号（防止亚稳态）
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        sw_sync[0] <= 3'b000;
        sw_sync[1] <= 3'b000;
    end else begin
        sw_sync[0] <= sw_in;        // 第一级同步
        sw_sync[1] <= sw_sync[0];   // 第二级同步
    end
end

// 为每个拨码开关进行独立的消抖处理
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 3; i = i + 1) begin
            debounce_counter[i] <= 25'b0;
        end
        sw_stable <= 3'b000;
        sw_out <= 3'b000;
    end else begin
        for (i = 0; i < 3; i = i + 1) begin
            if (sw_sync[1][i] != sw_stable[i]) begin
                // 开关状态发生变化，开始消抖计数
                if (debounce_counter[i] < COUNTER_MAX) begin
                    debounce_counter[i] <= debounce_counter[i] + 1'b1;
                end else begin
                    // 消抖时间到，更新稳定状态
                    sw_stable[i] <= sw_sync[1][i];
                    sw_out[i] <= sw_sync[1][i];
                    debounce_counter[i] <= 25'b0;
                end
            end else begin
                // 开关状态稳定，重置计数器
                debounce_counter[i] <= 25'b0;
            end
        end
    end
end

endmodule