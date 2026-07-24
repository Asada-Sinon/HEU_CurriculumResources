
module led_controller (
    input clk,
    input clk_2Hz,
    input rst_n,
    input password_correct,
    input alarm_signal,
    input [2:0] current_bit_pos,
    input [2:0] current_state_debug,
    input [1:0] input_mode,           // 修改为2位：0=按键，1=键盘，2=蓝牙
    input [7:0] led_blink_ctrl,       // 新增：8位LED闪烁控制
    input blink_clock,                // 新增：闪烁时钟信号
    output reg [15:0] led
);

// LED控制逻辑
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        led <= 16'b0000000000000000;
    end else begin
        if (password_correct) begin
            // 密码正确：左半部分LED以2Hz频率闪烁
            if (clk_2Hz) begin
                led <= 16'b1111111100000000;
            end else begin
                led <= 16'b0000000000000000;
            end
        end else if (alarm_signal) begin
            // 报警：右半部分LED以2Hz频率闪烁
            if (clk_2Hz) begin
                led <= 16'b0000000011111111;
            end else begin
                led <= 16'b0000000000000000;
            end
        end else if (current_state_debug == 3'b001) begin
            // 设置模式：显示当前位位置（8位对应LED8-LED15）
            led[15:8] <= 8'b00000000; // 先清零高8位
            led[7:0] <= 8'b00000000;  // 先清零低8位
            
            // 根据当前位位置点亮对应LED
            case (current_bit_pos)
                3'd0: led[15] <= 1'b1; // LED15亮（第0位，BIT1）
                3'd1: led[14] <= 1'b1; // LED14亮（第1位，BIT2）
                3'd2: led[13] <= 1'b1; // LED13亮（第2位，BIT3）
                3'd3: led[12] <= 1'b1; // LED12亮（第3位，BIT4）
                3'd4: led[11] <= 1'b1; // LED11亮（第4位，BIT5）
                3'd5: led[10] <= 1'b1; // LED10亮（第5位，BIT6）
                3'd6: led[9] <= 1'b1;  // LED9亮（第6位，BIT7）
                3'd7: led[8] <= 1'b1;  // LED8亮（第7位，BIT8）
            endcase
            
            // 使用新的闪烁控制信号进行闪烁
            if (led_blink_ctrl[current_bit_pos] && !blink_clock) begin
                // 闪烁的熄灭状态，清除对应位的LED
                case (current_bit_pos)
                    3'd0: led[15] <= 1'b0;
                    3'd1: led[14] <= 1'b0;
                    3'd2: led[13] <= 1'b0;
                    3'd3: led[12] <= 1'b0;
                    3'd4: led[11] <= 1'b0;
                    3'd5: led[10] <= 1'b0;
                    3'd6: led[9] <= 1'b0;
                    3'd7: led[8] <= 1'b0;
                endcase
            end
        end else if (current_state_debug == 3'b010) begin
            // 输入模式：显示当前位位置（支持所有输入模式的闪烁）
            led[15:8] <= 8'b00000000; // 先清零高8位
            led[7:0] <= 8'b00000000;  // 先清零低8位
            
            // 根据当前位位置点亮对应LED
            case (current_bit_pos)
                3'd0: led[15] <= 1'b1; // LED15亮（第0位，BIT1）
                3'd1: led[14] <= 1'b1; // LED14亮（第1位，BIT2）
                3'd2: led[13] <= 1'b1; // LED13亮（第2位，BIT3）
                3'd3: led[12] <= 1'b1; // LED12亮（第3位，BIT4）
                3'd4: led[11] <= 1'b1; // LED11亮（第4位，BIT5）
                3'd5: led[10] <= 1'b1; // LED10亮（第5位，BIT6）
                3'd6: led[9] <= 1'b1;  // LED9亮（第6位，BIT7）
                3'd7: led[8] <= 1'b1;  // LED8亮（第7位，BIT8）
            endcase
            
            // 使用新的闪烁控制信号进行闪烁（所有输入模式都支持）
            if (led_blink_ctrl[current_bit_pos] && !blink_clock) begin
                // 闪烁的熄灭状态，清除对应位的LED
                case (current_bit_pos)
                    3'd0: led[15] <= 1'b0;
                    3'd1: led[14] <= 1'b0;
                    3'd2: led[13] <= 1'b0;
                    3'd3: led[12] <= 1'b0;
                    3'd4: led[11] <= 1'b0;
                    3'd5: led[10] <= 1'b0;
                    3'd6: led[9] <= 1'b0;
                    3'd7: led[8] <= 1'b0;
                endcase
            end
        end else begin
            // 其他状态：全部关闭
            led <= 16'b0000000000000000;
        end
    end
end

endmodule