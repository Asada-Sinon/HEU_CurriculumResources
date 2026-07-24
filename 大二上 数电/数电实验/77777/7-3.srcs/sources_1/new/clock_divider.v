`timescale 1ns / 1ps

//==============================================================================
// 时钟分频模块 - 修正为100MHz输入
//==============================================================================
module clock_divider (
    input clk_in,        // 100MHz输入时钟
    input rst_n,         // 复位信号
    output reg clk_2Hz   // 2Hz输出时钟
);

// 分频计数器 - 修正为100MHz
reg [25:0] count;

parameter COUNT_MAX = 26'd25000000; // 100MHz / 50M = 2Hz

always @(posedge clk_in or negedge rst_n) begin
    if (!rst_n) begin
        count <= 26'b0;
        clk_2Hz <= 1'b0;
    end else begin
        if (count >= COUNT_MAX - 1) begin
            count <= 26'b0;
            clk_2Hz <= ~clk_2Hz;
        end else begin
            count <= count + 1'b1;
        end
    end
end

endmodule