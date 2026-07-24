//串口指令发�?�模�?
module uart_data_tx(
    input clk,    //时钟
    input rst_n,  //复位


	input tx_en,      //命令发�?�使�?   
	input [7:0]tx_num, //本次命令发�?�的字节�?
	input [799:0]tx_data, //本次命令发�?�的数据


	input [2:0]Baud_Set,   //波特率设�?
	output  reg uart_tx_done,   //本次指令发�?�完�?

	output  uart_tx   //串口发�?�引�?
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

//串口大单字节发�?�模�?
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

	input Clk ;    //模块全局时钟输入�?50M
	input Rst_n;    //复位信号输入，低有效
	input [7:0]data_byte;  //待传�?8bit数据
	input send_en;    //发�?�使�?
	input [2:0]Baud_Set;   //波特率设�?
	
	output reg uart_tx;    //串口输出信号
	output reg Tx_Done;    //1byte数据发�?�完成标�?
	output reg uart_state; //发�?�数据状�?

	localparam START_BIT = 1'b0;
	localparam STOP_BIT = 1'b1; 
	
	reg bps_clk;	     //波特率时�?	
	reg [15:0]div_cnt;      //分频计数�?	
	reg [15:0]bps_DR;       //分频计数�?大�??	
	reg [3:0]bps_cnt;      //波特率时钟计数器	
	reg [7:0]data_byte_reg;//data_byte寄存后数�?
	
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
	
	//counter
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
