// 按键消抖模块
module key_debounce (
    input clk,
    input rst_n,
    input key_in,               // 按键输入
    output reg key_pressed      // 按键按下脉冲输出
);

// 消抖时间20ms@100MHz
parameter DEBOUNCE_TIME = 20'd2_000_000;

reg [19:0] debounce_cnt;
reg key_reg1, key_reg2, key_reg3;
reg key_state;

// 按键同步处理
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        key_reg1 <= 1'b0;
        key_reg2 <= 1'b0;
        key_reg3 <= 1'b0;
    end else begin
        key_reg1 <= key_in;
        key_reg2 <= key_reg1;
        key_reg3 <= key_reg2;
    end
end

// 按键消抖和边沿检测
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        debounce_cnt <= 20'd0;
        key_state <= 1'b0;
        key_pressed <= 1'b0;
    end else begin
        key_pressed <= 1'b0;  // 默认为0
        
        if (key_reg2 == key_reg3) begin
            if (debounce_cnt >= DEBOUNCE_TIME - 1) begin
                if (key_state == 1'b0 && key_reg3 == 1'b1) begin
                    key_pressed <= 1'b1;  // 检测到上升沿
                end
                key_state <= key_reg3;
                debounce_cnt <= 20'd0;
            end else begin
                debounce_cnt <= debounce_cnt + 1'b1;
            end
        end else begin
            debounce_cnt <= 20'd0;
        end
    end
end

endmodule