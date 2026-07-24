`timescale 1ns / 1ps

//==============================================================================
// 4×4矩阵键盘扫描模块 - 参考映射版本
//==============================================================================
module keyboard_4x4 (
    input clk,                      // 100MHz系统时钟
    input reset_n,                  // 复位信号
    input [3:0] keyboard_row_x4_i,  // 4行输入
    output reg [3:0] keyboard_col_x4_o, // 4列输出
    output reg key_flag,            // 按键标志
    output reg [3:0] key_value      // 按键值
);

// 扫描时钟分频 - 修正为100MHz
reg [16:0] scan_counter;
reg scan_clk;

always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        scan_counter <= 17'b0;
        scan_clk <= 1'b0;
    end else begin
        if (scan_counter >= 17'd50000) begin // 100MHz / 50000 = 2kHz
            scan_counter <= 17'b0;
            scan_clk <= ~scan_clk;
        end else begin
            scan_counter <= scan_counter + 1'b1;
        end
    end
end

// 列扫描状态
reg [1:0] col_scan;
reg key_detected;
reg [3:0] detected_key;

// 消抖相关 - 修正消抖时间
reg [15:0] debounce_counter;
reg key_pressed_last;
reg key_pressed_stable;

// 扫描逻辑 - 按照参考代码的映射
always @(posedge scan_clk or negedge reset_n) begin
    if (!reset_n) begin
        col_scan <= 2'b00;
        keyboard_col_x4_o <= 4'b0111; // 从C1开始
        key_detected <= 1'b0;
        detected_key <= 4'hF;
    end else begin
        // 列扫描 - 按照参考代码的列输出模式
        case (col_scan)
            2'b00: keyboard_col_x4_o <= 4'b0111; // C1列
            2'b01: keyboard_col_x4_o <= 4'b1011; // C2列
            2'b10: keyboard_col_x4_o <= 4'b1101; // C3列
            2'b11: keyboard_col_x4_o <= 4'b1110; // C4列
        endcase
        
        // 按键检测 - 完全按照参考代码的映射
        key_detected <= 1'b0;
        detected_key <= 4'hF;
        
        case (col_scan)
            2'b00: begin // C1列(4'b0111)
                case (keyboard_row_x4_i)
                    4'b0111: begin // R1行
                        key_detected <= 1'b1;
                        detected_key <= 4'h1; // 按键1
                    end
                    4'b1011: begin // R2行
                        key_detected <= 1'b1;
                        detected_key <= 4'h4; // 按键4
                    end
                    4'b1101: begin // R3行
                        key_detected <= 1'b1;
                        detected_key <= 4'h7; // 按键7
                    end
                    4'b1110: begin // R4行
                        key_detected <= 1'b1;
                        detected_key <= 4'h0; // 按键0
                    end
                endcase
            end
            
            2'b01: begin // C2列(4'b1011)
                case (keyboard_row_x4_i)
                    4'b0111: begin // R1行
                        key_detected <= 1'b1;
                        detected_key <= 4'h2; // 按键2
                    end
                    4'b1011: begin // R2行
                        key_detected <= 1'b1;
                        detected_key <= 4'h5; // 按键5
                    end
                    4'b1101: begin // R3行
                        key_detected <= 1'b1;
                        detected_key <= 4'h8; // 按键8
                    end
                    4'b1110: begin // R4行
                        key_detected <= 1'b1;
                        detected_key <= 4'hF; // 按键F (#键)
                    end
                endcase
            end
            
            2'b10: begin // C3列(4'b1101)
                case (keyboard_row_x4_i)
                    4'b0111: begin // R1行
                        key_detected <= 1'b1;
                        detected_key <= 4'h3; // 按键3
                    end
                    4'b1011: begin // R2行
                        key_detected <= 1'b1;
                        detected_key <= 4'h6; // 按键6
                    end
                    4'b1101: begin // R3行
                        key_detected <= 1'b1;
                        detected_key <= 4'h9; // 按键9
                    end
                    4'b1110: begin // R4行
                        key_detected <= 1'b1;
                        detected_key <= 4'hE; // 按键E (*键)
                    end
                endcase
            end
            
            2'b11: begin // C4列(4'b1110)
                case (keyboard_row_x4_i)
                    4'b0111: begin // R1行
                        key_detected <= 1'b1;
                        detected_key <= 4'hA; // 按键A
                    end
                    4'b1011: begin // R2行
                        key_detected <= 1'b1;
                        detected_key <= 4'hB; // 按键B
                    end
                    4'b1101: begin // R3行
                        key_detected <= 1'b1;
                        detected_key <= 4'hC; // 按键C
                    end
                    4'b1110: begin // R4行
                        key_detected <= 1'b1;
                        detected_key <= 4'hD; // 按键D
                    end
                endcase
            end
        endcase
        
        // 更新扫描列
        col_scan <= col_scan + 1'b1;
    end
end

// 消抖处理 - 修正为100MHz时钟
always @(posedge clk or negedge reset_n) begin
    if (!reset_n) begin
        debounce_counter <= 16'b0;
        key_pressed_last <= 1'b0;
        key_pressed_stable <= 1'b0;
        key_flag <= 1'b0;
        key_value <= 4'hF;
    end else begin
        key_pressed_last <= key_detected;
        
        // 检测按键状态变化
        if (key_detected != key_pressed_last) begin
            debounce_counter <= 16'b0;
            key_pressed_stable <= key_pressed_last;
        end else begin
            if (debounce_counter < 16'd10000) begin // 100ms消抖时间 @ 100MHz
                debounce_counter <= debounce_counter + 1'b1;
            end else begin
                key_pressed_stable <= key_detected;
                
                // 生成按键标志（上升沿检测）
                if (key_detected && !key_pressed_stable && detected_key != 4'hF) begin
                    key_flag <= 1'b1;
                    key_value <= detected_key;
                end else begin
                    key_flag <= 1'b0;
                end
            end
        end
    end
end

endmodule