//每次从字符串中一次取出一个字节，然后启动 uart_byte_tx 模块去发送这个字节
//uart_byte_tx 发送完成后，它会返回一个完成信号，然后 uart_data_tx 再去取下一个字节
module uart_data_tx(
    input clk,   
    input rst_n,  
	input tx_en,      //发送使能
	input [7:0]tx_num, //发送字节总数
	input [799:0]tx_data, //数据缓存区域
	input [2:0]Baud_Set,   
	output  reg uart_tx_done,   
	output  uart_tx   
);
wire byte_tx_done;  
reg byte_tx_en;
reg [7:0]byte_tx_cnt;
reg [7:0]byte_tx_data;
reg [1:0]state;
always@(posedge clk or negedge rst_n)
if(~rst_n)
begin
	state <= 'h0;
	byte_tx_en <= 1'b0;
	byte_tx_cnt <= 0;
	uart_tx_done <= 1'b0;
end
else 
begin
	case(state)
	0:
	begin
		uart_tx_done <= 1'b0;
		if(tx_en)
			state <= 1;
	end
	1:
	begin
		state <= 2;
		byte_tx_en <= 1'b1;
		byte_tx_data<= tx_data[((tx_num-byte_tx_cnt)*8-1)-:8];
	end
	2:
	begin
		byte_tx_en <= 1'b0;
		if(byte_tx_done)
		begin
			if(byte_tx_cnt==tx_num-1)
			begin
				byte_tx_cnt <= 0;
				uart_tx_done <= 1'b1;
				state <= 0;
			end
			else begin
				byte_tx_cnt <= byte_tx_cnt+1;
				state <= 1;
			end
		end
	end
	default:state <= 0;
	endcase
end
	uart_byte_tx uart_byte_tx(
		.Clk(clk),
		.Rst_n(rst_n),
		.data_byte(byte_tx_data),
		.send_en(byte_tx_en),   
		.Baud_Set(Baud_Set),  
		.uart_tx(uart_tx),  
		.Tx_Done(byte_tx_done),   
		.uart_state(uart_state) 
	);
endmodule
//最底层的物理发送
module uart_byte_tx(
	Clk,
	Rst_n,
  
	data_byte,
	send_en,   
	Baud_Set,  
	
	uart_tx,  
	Tx_Done,   
	uart_state 
);
	input Clk ;    
	input Rst_n;   
	input [7:0]data_byte;  
	input send_en;    
	input [2:0]Baud_Set;  
	output reg uart_tx;    
	output reg Tx_Done;   
	output reg uart_state;
	localparam START_BIT = 1'b0;
	localparam STOP_BIT = 1'b1; 
	reg bps_clk;	     
	reg [15:0]div_cnt;     
	reg [15:0]bps_DR;       
	reg [3:0]bps_cnt;      
	reg [7:0]data_byte_reg;
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		uart_state <= 1'b0;
	else if(send_en)
		uart_state <= 1'b1;
	else if(bps_cnt == 4'd11)
		uart_state <= 1'b0;
	else
		uart_state <= uart_state;
	
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		data_byte_reg <= 8'd0;
	else if(send_en)
		data_byte_reg <= data_byte;
	else
		data_byte_reg <= data_byte_reg;
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		bps_DR <= 16'd5207;
	else begin
		case(Baud_Set)
			0:bps_DR <= 16'd5207;
			1:bps_DR <= 16'd2603;
			2:bps_DR <= 16'd1301;
			3:bps_DR <= 16'd867;
			4:bps_DR <= 16'd433;
			default:bps_DR <= 16'd5207;			
		endcase
	end	
	//div_cnt 完整计数一次需要 10414 + 1 = 10415 个 Clk 周期
	//9600波特率
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		div_cnt <= 16'd0;
	else if(uart_state)begin
		if(div_cnt == bps_DR*2)
			div_cnt <= 16'd0;
		else
			div_cnt <= div_cnt + 1'b1;
	end
	else
		div_cnt <= 16'd0;
	
	// bps_clk gen
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		bps_clk <= 1'b0;
	else if(div_cnt == 16'd1)
		bps_clk <= 1'b1;
	else
		bps_clk <= 1'b0;
	
	//bps counter
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)	
		bps_cnt <= 4'd0;
	else if(bps_cnt == 4'd11)
		bps_cnt <= 4'd0;
	else if(bps_clk)
		bps_cnt <= bps_cnt + 1'b1;
	else
		bps_cnt <= bps_cnt;
		
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		Tx_Done <= 1'b0;
	else if(bps_cnt == 4'd11)
		Tx_Done <= 1'b1;
	else
		Tx_Done <= 1'b0;
		
	always@(posedge Clk or negedge Rst_n)
	if(!Rst_n)
		uart_tx <= 1'b1;
	else begin
		case(bps_cnt)
			0:uart_tx <= 1'b1;
			1:uart_tx <= START_BIT;
			2:uart_tx <= data_byte_reg[0];
			3:uart_tx <= data_byte_reg[1];
			4:uart_tx <= data_byte_reg[2];
			5:uart_tx <= data_byte_reg[3];
			6:uart_tx <= data_byte_reg[4];
			7:uart_tx <= data_byte_reg[5];
			8:uart_tx <= data_byte_reg[6];
			9:uart_tx <= data_byte_reg[7];
			10:uart_tx <= STOP_BIT;
			default:uart_tx <= 1'b1;
		endcase
	end	

endmodule
