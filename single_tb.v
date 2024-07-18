`timescale 1ns / 1ps


module testbench;
	reg clk;
	reg reset;
	wire [31:0] WriteData;
	wire [31:0] ALUResult;
	wire MemWrite;
	wire [31:0] PC;
	wire [31:0] Instr;
	wire [31:0] ReadData;
	wire ALUFlags ;
	wire CondEx ; 
	top dut(
		.clk(clk),
		.reset(reset),
		.WriteData(WriteData),
		.DataAdr(ALUResult),
		.MemWrite(MemWrite) , 
		.PC(PC) ,
		.Instr(Instr),
		 .ReadData(ReadData),
		 .ALUFlags(ALUFlags) , 
		 .CondEx(CondEx)
	);
	initial begin
		reset <= 1;
		#(22)
			;
		reset <= 0;
	end
	always begin
		clk <= 1;
		#(5)
			;
		clk <= 0;
		#(5)
			;
	end

endmodule
