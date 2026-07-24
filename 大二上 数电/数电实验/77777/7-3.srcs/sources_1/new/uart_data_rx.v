`timescale 1ns / 1ps

module uart_data_rx(
    input clk,
    input rst_n,
    input uart_rx,
    input [2:0] Baud_Set,
    output reg [7:0] rx_data,
    output reg rx_done
);

// 波特率设置参数
parameter BAUD_9600   = 10416; // 100MHz/9600
parameter BAUD_19200  = 5208;  // 100MHz/19200
parameter BAUD_38400  = 2604;  // 100MHz/38400
parameter BAUD_115200 = 868;   // 100MHz/115200

reg [15:0] baud_cnt;
reg [15:0] baud_max;
reg [3:0] bit_cnt;
reg [7:0] rx_reg;
reg rx_flag;
reg uart_rx_d1, uart_rx_d2;

// 波特率选择
always@(*) begin
    case(Baud_Set)
        3'd0: baud_max = BAUD_9600;
        3'd1: baud_max = BAUD_19200;
        3'd2: baud_max = BAUD_38400;
        3'd3: baud_max = BAUD_115200;
        default: baud_max = BAUD_9600;
    endcase
end

// 输入同步
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        uart_rx_d1 <= 1'b1;
        uart_rx_d2 <= 1'b1;
    end
    else begin
        uart_rx_d1 <= uart_rx;
        uart_rx_d2 <= uart_rx_d1;
    end
end

wire rx_negedge = uart_rx_d2 & (~uart_rx_d1);

// 接收状态机
always@(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        baud_cnt <= 16'd0;
        bit_cnt <= 4'd0;
        rx_flag <= 1'b0;
        rx_reg <= 8'd0;
        rx_data <= 8'd0;
        rx_done <= 1'b0;
    end
    else begin
        rx_done <= 1'b0;
        
        if(rx_negedge && !rx_flag) begin
            rx_flag <= 1'b1;
            baud_cnt <= 16'd0;
            bit_cnt <= 4'd0;
        end
        else if(rx_flag) begin
            if(baud_cnt == baud_max - 1) begin
                baud_cnt <= 16'd0;
                bit_cnt <= bit_cnt + 1'b1;
                
                case(bit_cnt)
                    4'd0: ; // 起始位，不处理
                    4'd1: rx_reg[0] <= uart_rx_d2;
                    4'd2: rx_reg[1] <= uart_rx_d2;
                    4'd3: rx_reg[2] <= uart_rx_d2;
                    4'd4: rx_reg[3] <= uart_rx_d2;
                    4'd5: rx_reg[4] <= uart_rx_d2;
                    4'd6: rx_reg[5] <= uart_rx_d2;
                    4'd7: rx_reg[6] <= uart_rx_d2;
                    4'd8: rx_reg[7] <= uart_rx_d2;
                    4'd9: begin // 停止位
                        rx_data <= rx_reg;
                        rx_done <= 1'b1;
                        rx_flag <= 1'b0;
                        bit_cnt <= 4'd0;
                    end
                endcase
            end
            else begin
                baud_cnt <= baud_cnt + 1'b1;
            end
        end
    end
end

endmodule