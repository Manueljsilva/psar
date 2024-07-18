module top (
	clk,
	reset,
	sel,
	vReg,
	sReg,
);
	input wire clk;
	input wire reset;
	wire [31:0] WriteData;
	wire [31:0] DataAdr;
	wire MemWrite;
	wire [31:0] PC;
	wire [31:0] Instr;
	wire [31:0] ReadData;
	wire [3:0] ALUFlags ;
	wire CondEx ; 

	input wire [3:0] sel;
	output wire [159:0] vReg;
	output wire [32:0] sReg;

	arm arm(
		.clk(clk),
		.reset(reset),
		.PC(PC),
		.Instr(Instr),
		.MemWrite(MemWrite),
		.ALUResult(DataAdr),
		.WriteData(WriteData),
		.ReadData(ReadData) , 
		.ALUFlags(ALUFlags) , 
		.CondEx(CondEx),
		.vReg(vReg),
		.sReg(sReg),
		.sel(sel)
	);
	imem imem(
		.a(PC),
		.rd(Instr)
	);
	dmem dmem(
		.clk(clk),
		.we(MemWrite),
		.a(DataAdr),
		.wd(WriteData),
		.rd(ReadData)
	);
endmodule
module dmem (
	clk,
	we,
	a,
	wd,
	rd
);
	input wire clk;
	input wire we;
	input wire [31:0] a;
	input wire [31:0] wd;
	output wire [31:0] rd;
	reg [31:0] RAM [63:0];
	assign rd = RAM[a[31:2]];
	always @(posedge clk)
		if (we)
			RAM[a[31:2]] <= wd;
endmodule
module imem (
	a,
	rd
);
	input wire [31:0] a;
	output wire [31:0] rd;
	reg [31:0] RAM [63:0];
	initial $readmemh("memfile.dat", RAM);
	assign rd = RAM[a[31:2]];
endmodule
module arm (
	clk,
	reset,
	PC,
	Instr,
	MemWrite,
	ALUResult,
	WriteData,
	ReadData , 
	ALUFlags , 
	CondEx,
	// Read register for display
	vReg,
	sReg,
	sel
);
	input wire clk;
	input wire reset;
	output wire [31:0] PC;
	input wire [31:0] Instr;
	output wire MemWrite;
	output wire [31:0] ALUResult;
	output wire [31:0] WriteData;
	input wire [31:0] ReadData;
	output wire [3:0] ALUFlags;
	wire RegWrite;
	wire ALUSrc;
	wire MemtoReg;
	wire PCSrc;
	wire [1:0] RegSrc;
	wire [1:0] ImmSrc;
	wire [2:0] ALUControl;
	output wire CondEx ; 

	// Floating point operations
	wire IsFP;

	// Vectorial operations
	wire IsV;
	wire RegWriteValue;

	// Read register for display
	output wire [159:0] vReg;
	output wire [32:0] sReg;
	input wire [3:0] sel;

	controller c(
		.clk(clk),
		.reset(reset),
		.Instr(Instr[31:12]),
		.ALUFlags(ALUFlags),
		.RegSrc(RegSrc),
		.RegWrite(RegWrite),
		.ImmSrc(ImmSrc),
		.ALUSrc(ALUSrc),
		.ALUControl(ALUControl),
		.MemWrite(MemWrite),
		.MemtoReg(MemtoReg),
		.PCSrc(PCSrc),
		.CondEx(CondEx),
		.IsFP(IsFP),
		.IsV(IsV),
		.RegWriteValue(RegWriteValue)
	);
	datapath dp(
		.clk(clk),
		.reset(reset),
		.RegSrc(RegSrc),
		.RegWrite(RegWrite),
		.ImmSrc(ImmSrc),
		.ALUSrc(ALUSrc),
		.ALUControl(ALUControl),
		.MemtoReg(MemtoReg),
		.PCSrc(PCSrc),
		.ALUFlags(ALUFlags),
		.PC(PC),
		.Instr(Instr),
		.ALUResult(ALUResult),
		.WriteData(WriteData),
		.ReadData(ReadData),
		// Floating point operations
		.IsFP(IsFP),
		.IsV(IsV),
		.RegWriteValue(RegWriteValue),
		// Read register for display
		.vReg(vReg),
		.sReg(sReg),
		.sel(sel)
	);
endmodule
module controller (
	clk,
	reset,
	Instr,
	ALUFlags,
	RegSrc,
	RegWrite,
	ImmSrc,
	ALUSrc,
	ALUControl,
	MemWrite,
	MemtoReg,
	PCSrc,
	CondEx,
	IsFP,
	IsV,
	RegWriteValue
);
	input wire clk;
	input wire reset;
	input wire [31:12] Instr;
	input wire [3:0] ALUFlags;
	output wire [1:0] RegSrc;
	output wire RegWrite;
	output wire [1:0] ImmSrc;
	output wire ALUSrc;
	output wire [2:0] ALUControl;
	output wire MemWrite;
	output wire MemtoReg;
	output wire PCSrc;
	wire [1:0] FlagW;
	wire PCS;
	wire RegW;
	wire MemW;
	output wire CondEx;

	// Floating point operations
	output wire IsFP;

	// Vectorial operations
	output wire IsV;
	output wire RegWriteValue;

	decode dec(
		.Op(Instr[27:26]),
		.Funct(Instr[25:20]),
		.Rd(Instr[15:12]),
		.FlagW(FlagW),
		.PCS(PCS),
		.RegW(RegW),
		.MemW(MemW),
		.MemtoReg(MemtoReg),
		.ALUSrc(ALUSrc),
		.ImmSrc(ImmSrc),
		.RegSrc(RegSrc),
		.ALUControl(ALUControl),
		.IsFP(IsFP),
		.IsV(IsV),
		.RegWriteValue(RegWriteValue)
	);
	condlogic cl(
		.clk(clk),
		.reset(reset),
		.Cond(Instr[31:28]),
		.ALUFlags(ALUFlags),
		.FlagW(FlagW),
		.PCS(PCS),
		.RegW(RegW),
		.MemW(MemW),
		.PCSrc(PCSrc),
		.RegWrite(RegWrite),
		.MemWrite(MemWrite) , 
		.CondEx(CondEx)
	);
endmodule
module decode (
	Op,
	Funct,
	Rd,
	FlagW,
	PCS,
	RegW,
	MemW,
	MemtoReg,
	ALUSrc,
	ImmSrc,
	RegSrc,
	ALUControl,
	IsFP,
	IsV,
	RegWriteValue
);
	input wire [1:0] Op;
	input wire [5:0] Funct;
	input wire [3:0] Rd;
	output reg [1:0] FlagW;
	output wire PCS;
	output wire RegW;
	output wire MemW;
	output wire MemtoReg;
	output wire ALUSrc;
	output wire [1:0] ImmSrc;
	output wire [1:0] RegSrc;
	output reg [2:0] ALUControl;
	reg [9:0] controls;
	wire Branch;
	wire ALUOp;

	// Floating point operations
	output reg IsFP;

	// Vectorial operations
	output reg IsV;
	output reg RegWriteValue;

	always @(*) begin
		casex (Op)
			2'b00:
				if (Funct[5])
					controls = 10'b0000101001;
				else
					controls = 10'b0000001001;
			2'b01:
				if (Funct[0])
					controls = 10'b0001111000;
				else
					controls = 10'b1001110100;
			2'b10: controls = 10'b0110100010;
			default: controls = 10'bxxxxxxxxxx;
		endcase
	end
	assign {RegSrc, ImmSrc, ALUSrc, MemtoReg, RegW, MemW, Branch, ALUOp} = controls;
	
	always @(*)
		if (ALUOp) begin
			RegWriteValue = 0;

			case (Funct[4:1])
				// Scalar operations
				// Add
				4'b0000: ALUControl = 3'b000;
				// Substract
				4'b0001: ALUControl = 3'b001;
				// AND
				4'b0010: ALUControl = 3'b010;
				// OR
				4'b0011: ALUControl = 3'b011;

				// Floating point operations
				// Single precision floating point add
				4'b0100: ALUControl = 3'b000; 
				// Half precision floating point add
				4'b0101: ALUControl = 3'b101;
				// Single precision floating point multiply
				4'b0110: ALUControl = 3'b100;
				// Half precision floating point multiply
				4'b0111: ALUControl = 3'b110;

				// Vectorial operations
				// Vectorial add (VADD)
				4'b1000: ALUControl = 3'b000;
				// Vectorial substract (VSUB)
				4'b1001: ALUControl = 3'b001;
				// Vectorial multiply (VMUL)
				4'b1010: ALUControl = 3'b100;
				// Vectorial AND (VAND)
				4'b1011: ALUControl = 3'b010;
				// Vectorial OR (VOR)
				4'b1100: ALUControl = 3'b011;
				// Vectorial XOR (VXOR)
				4'b1101: ALUControl = 3'b111;
				// Vectorial floating point add (VADDFP)
				4'b1110: ALUControl = 3'b000;
				
				// Vectorial MOV for 1 value 
				4'b1111: begin
					RegWriteValue = 1;
					ALUControl = 3'bxxx;
				end

				default: ALUControl = 3'bxxx;
			endcase
			FlagW[1] = Funct[0];
			// TODO: Update with the news cmd
			FlagW[0] = Funct[0] & ((ALUControl == 3'b000) | (ALUControl == 3'b001));

            // Funct[0] & ((ALUControl == 3'b000) | (ALUControl == 3'b001)) = Funct[3] & (~Funct[4] | (Funct[2] & ~Funct[1]))
			IsFP = Funct[3] & (~Funct[4] | (Funct[2] & ~Funct[1]));
			IsV = Funct[4];
		end
		else begin
			ALUControl = 3'b000;
			FlagW = 2'b00;
			IsFP = 0;
			IsV = 0;
		end
	assign PCS = ((Rd == 4'b1111) & RegW) | Branch;
endmodule
module condlogic (
	clk,
	reset,
	Cond,
	ALUFlags,
	FlagW,
	PCS,
	RegW,
	MemW,
	PCSrc,
	RegWrite,
	MemWrite,
	CondEx 
);
	input wire clk;
	input wire reset;
	input wire [3:0] Cond;
	input wire [3:0] ALUFlags;
	input wire [1:0] FlagW;
	input wire PCS;
	input wire RegW;
	input wire MemW;
	output wire PCSrc;
	output wire RegWrite;
	output wire MemWrite;
	wire [1:0] FlagWrite;
	wire [3:0] Flags;
	output wire CondEx;
	flopenr #(2) flagreg1(
		.clk(clk),
		.reset(reset),
		.en(FlagWrite[1]),
		.d(ALUFlags[3:2]),
		.q(Flags[3:2])
	);
	flopenr #(2) flagreg0(
		.clk(clk),
		.reset(reset),
		.en(FlagWrite[0]),
		.d(ALUFlags[1:0]),
		.q(Flags[1:0])
	);
	condcheck cc(
		.Cond(Cond),
		.Flags(Flags),
		.CondEx(CondEx)
	);
	assign FlagWrite = FlagW & {2 {CondEx}};
	assign RegWrite = RegW & CondEx;
	assign MemWrite = MemW & CondEx;
	assign PCSrc = PCS & CondEx;
endmodule
module condcheck (
	Cond,
	Flags,
	CondEx
);
	input wire [3:0] Cond;
	input wire [3:0] Flags;
	output reg CondEx;
	wire neg;
	wire zero;
	wire carry;
	wire overflow;
	wire ge;
	assign {neg, zero, carry, overflow} = Flags;
	assign ge = neg == overflow;
	always @(*)
		case (Cond)
			4'b0000: CondEx = zero;
			4'b0001: CondEx = ~zero;
			4'b0010: CondEx = carry;
			4'b0011: CondEx = ~carry;
			4'b0100: CondEx = neg;
			4'b0101: CondEx = ~neg;
			4'b0110: CondEx = overflow;
			4'b0111: CondEx = ~overflow;
			4'b1000: CondEx = carry & ~zero;
			4'b1001: CondEx = ~(carry & ~zero);
			4'b1010: CondEx = ge;
			4'b1011: CondEx = ~ge;
			4'b1100: CondEx = ~zero & ge;
			4'b1101: CondEx = ~(~zero & ge);
			4'b1110: CondEx = 1'b1;
			default: CondEx = 1'bx;
		endcase
endmodule

module datapath (
	clk,
	reset,
	RegSrc,
	RegWrite,
	ImmSrc,
	ALUSrc,
	ALUControl,
	MemtoReg,
	PCSrc,
	ALUFlags,
	PC,
	Instr,
	ALUResult,
	WriteData,
	ReadData,
	// Floating point operations
	IsFP,
	// Vectorial operations
	IsV,
	RegWriteValue,
	// Read register for display
	vReg,
	sReg,
	sel
);
	input wire clk;
	input wire reset;
	input wire [1:0] RegSrc;
	input wire RegWrite;
	// RegWriteValue for vectorial operations
	input wire RegWriteValue;
	input wire [1:0] ImmSrc;
	input wire ALUSrc;
	input wire [2:0] ALUControl;
	input wire MemtoReg;
	input wire PCSrc;
	output wire [3:0] ALUFlags;
	output wire [31:0] PC;
	input wire [31:0] Instr;
	output wire [31:0] ALUResult;
	output wire [31:0] WriteData;
	input wire [31:0] ReadData;
	wire [31:0] PCNext;
	wire [31:0] PCPlus4;
	wire [31:0] PCPlus8;
	wire [31:0] ExtImm;
	wire [31:0] SrcA;
	wire [31:0] SrcB;
	wire [31:0] Result;
	wire [3:0] RA1;
	wire [3:0] RA2;

	// Floating point operations
	input wire IsFP;

	// Vectorial operations
	input wire IsV;

	// Read register for display
	output wire [159:0] vReg;
	output wire [32:0] sReg;
	input wire [3:0] sel;

	mux2 #(32) pcmux(
		.d0(PCPlus4),
		.d1(Result),
		.s(PCSrc),
		.y(PCNext)
	);
	flopr #(32) pcreg(
		.clk(clk),
		.reset(reset),
		.d(PCNext),
		.q(PC)
	);
	adder #(32) pcadd1(
		.a(PC),
		.b(32'b100),
		.y(PCPlus4)
	);
	adder #(32) pcadd2(
		.a(PCPlus4),
		.b(32'b100),
		.y(PCPlus8)
	);
	mux2 #(4) ra1mux(
		.d0(Instr[19:16]),
		.d1(4'b1111),
		.s(RegSrc[0]),
		.y(RA1)
	);
	mux2 #(4) ra2mux(
		.d0(Instr[3:0]),
		.d1(Instr[15:12]),
		.s(RegSrc[1]),
		.y(RA2)
	);
	regfile rf(
		.clk(clk),
		// If we and its not a vectorial operation
		.we3(RegWrite & ~IsV),
		.ra1(RA1),
		.ra2(RA2),
		.wa3(Instr[15:12]),
		.wd3(Result),
		.r15(PCPlus8),
		.rd1(SrcA),
		.rd2(WriteData),
		// Read register for display
		.sel(sel),
		.sReg(sReg)
	);
	mux2 #(32) resmux(
		.d0(ALUResult),
		.d1(ReadData),
		.s(MemtoReg),
		.y(Result)
	);
    extend ext(
		.Instr(Instr[23:0]),
		.ImmSrc(ImmSrc),
		.ExtImm(ExtImm)
	);
	wire [31:0] ExtImmRot;
	    // rot a src2
    rot rot(
        .b(ExtImm),
        .Rot(Instr[11:8]),
        .Result(ExtImmRot)
    );
    
	mux2 #(32) srcbmux(
		.d0(WriteData),
		.d1(ExtImmRot),
		.s(ALUSrc),
		.y(SrcB)
	);

    
	alu alu(
		.a(SrcA),
		.b(SrcB),
		.ALUControl(ALUControl),
		.Result(ALUResult),
		.ALUFlags(ALUFlags),
		.IsFP(IsFP) 
	);

    // Vectorial operations
	// Vectorial register file
    wire [159:0] VSrcA;
    wire [159:0] VSrcB;
    wire [159:0] VResult;

	Valu alu_v(
		.Va(VSrcA),
		.Vb(VSrcB),
		.ALUControl(ALUControl),
		.VResult(VResult),
		.IsFP(IsFP) 
	);

  VRegFile vrf(
		.clk(clk),
		// If we and its a vectorial operation
		.vector_we(RegWrite & IsV),
		.vector_wev(RegWriteValue),
		.vector_ra1(Instr[19:16]),
		.vector_ra2(RA2),
		.vector_wa(Instr[15:12]),
		.vector_wd(VResult),
		.vector_wv(SrcB),
		.vector_rd1(VSrcA),
		.vector_rd2(VSrcB),
		// Read register for display
		.sel(sel),
		.vReg(vReg)
	);
endmodule

module regfile (
	clk,
	we3,
	ra1,
	ra2,
	wa3,
	wd3,
	r15,
	rd1,
	rd2,
	// Read register for display
	sel,
	sReg,
	sel
);
	input wire clk;
	input wire we3;
	input wire [3:0] ra1;
	input wire [3:0] ra2;
	input wire [3:0] wa3;
	input wire [31:0] wd3;
	input wire [31:0] r15;
	output wire [31:0] rd1;
	output wire [31:0] rd2;
	reg [31:0] rf [14:0];
	always @(posedge clk)
		if (we3)
			rf[wa3] <= wd3;
	assign rd1 = (ra1 == 4'b1111 ? r15 : rf[ra1]);
	assign rd2 = (ra2 == 4'b1111 ? r15 : rf[ra2]);

	// Read register for display
	input wire [3:0] sel;
	output wire [32:0] sReg;
	assign sReg = rf[sel];
endmodule
module extend (
	Instr,
	ImmSrc,
	ExtImm
);
	input wire [23:0] Instr;
	input wire [1:0] ImmSrc;
	output reg [31:0] ExtImm;
	always @(*)
		case (ImmSrc)
			2'b00: ExtImm = {24'b000000000000000000000000, Instr[7:0]};
			2'b01: ExtImm = {20'b00000000000000000000, Instr[11:0]};
			2'b10: ExtImm = {{6 {Instr[23]}}, Instr[23:0], 2'b00};
			default: ExtImm = 32'bxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx;
		endcase
endmodule
module adder (
	a,
	b,
	y
);
	parameter WIDTH = 8;
	input wire [WIDTH - 1:0] a;
	input wire [WIDTH - 1:0] b;
	output wire [WIDTH - 1:0] y;
	assign y = a + b;
endmodule
module flopenr (
	clk,
	reset,
	en,
	d,
	q
);
	parameter WIDTH = 8;
	input wire clk;
	input wire reset;
	input wire en;
	input wire [WIDTH - 1:0] d;
	output reg [WIDTH - 1:0] q;
	always @(posedge clk or posedge reset)
		if (reset)
			q <= 0;
		else if (en)
			q <= d;
endmodule
module flopr (
	clk,
	reset,
	d,
	q
);
	parameter WIDTH = 8;
	input wire clk;
	input wire reset;
	input wire [WIDTH - 1:0] d;
	output reg [WIDTH - 1:0] q;
	always @(posedge clk or posedge reset)
		if (reset)
			q <= 0;
		else
			q <= d;
endmodule

module rot (
    input [31:0] b, // valor que rotará
    input [3:0] Rot, // cantidad de rotación
    output reg [31:0] Result // resultado de la rotación
);
    integer i;
	//rotacion a la izquierda
    always @* begin
        Result = b;
        for (i = 0; i < Rot; i = i + 1) begin
            Result = {Result[0], Result[31:1]};
        end
    end
endmodule



module alu(
		a, 
		b, 
		ALUControl, 
		Result, 
		ALUFlags,
		// Floating point operations
		IsFP 
	);
	input [31:0] a; 
	input [31:0] b;
	input [2:0] ALUControl;
    input wire IsFP;
	output wire [31:0] Result;
	output wire [3:0] ALUFlags;
    // Floating point operations
    wire [31:0] FPResult;

    // FPALU
    fp_alu fp_alu(
        .a(a),
        .b(b),
        .ALUControl(ALUControl),
        .Result(FPResult)
    );

    // Scalar operations
    wire [31:0] ScalarResult;
    // Scalar ALU
    scalar_alu scalar_alu(
        .a(a),
        .b(b),
        .ALUControl(ALUControl),
        .Result(ScalarResult),
        .ALUFlags(ALUFlags)
    );

    // Mux para seleccionar entre el resultado de la ALU normal y el de la ALU de punto flotante
    mux2 #(32) mux(
        .d0(ScalarResult),
        .d1(FPResult),
        .s(IsFP),
        .y(Result)
    );
endmodule

module scalar_alu(
		a, 
		b, 
		ALUControl, 
		Result, 
		ALUFlags
	);
    output [3:0] ALUFlags;
	input [31:0] a; 
	input [31:0] b;
	input [2:0] ALUControl;
	output reg [31:0] Result;

	wire neg, zero, carry, overflow;

	wire [31:0] condinvb;
	wire [32:0] sum;

	assign condinvb = ALUControl[0] ? ~b : b;
	assign sum = a + condinvb + ALUControl[0];

	always @(*)
			begin
				casex (ALUControl[2:0]) 
					3'b00?: Result = sum ; 
					3'b010: Result = a & b;
					3'b011: Result = a | b;
					3'b100: Result = a * b;
					3'b111: Result = a ^ b;
				endcase
			end

	assign neg = Result[31];
	assign zero = (Result == 32'b0);
	assign carry = (ALUControl[1] == 1'b0)  & sum[32];
	assign overflow = (ALUControl[1] == 1'b0) & ~(a[31] ^ b[31] ^ ALUControl[0]) & (a[31] ^ sum[31]);
	assign ALUFlags = {neg , zero , carry , overflow};
endmodule

module fp_alu(
    input [31:0] a,
    input [31:0] b,
    input [2:0] ALUControl,
    output reg [31:0] Result
);

    wire [31:0] fp32_adder_result;
    wire [31:0] fp16_adder_result;
    wire [31:0] fp32_mult_result;
    wire [31:0] fp16_mult_result;

	//extraer los 16 bits de a y b 
	wire [15:0] a16 = a[31:16];
	wire [15:0] b16 = b[31:16];

    // FP32 Adder
    fp_adder #(.EXP_WIDTH(8), .MAN_WIDTH(23), .BIAS(127)) fp32_adder (
        .a(a),
        .b(b),
        .result(fp32_adder_result)
    );

    // FP16 Adder (ajustar si necesitas diferentes entradas)
    fp_adder #(.EXP_WIDTH(5), .MAN_WIDTH(10), .BIAS(15)) fp16_adder (
		.a(a16),
        .b(b16),
        .result(fp16_adder_result)
    );

    // FP32 Multiplier (proporcionar módulo adecuado)
	fp_multiplier #(.EXP_WIDTH(8), .MAN_WIDTH(23), .BIAS(127)) fp32_mult (
		.a(a),
		.b(b),
		.result(fp32_mult_result)
	);

	// FP16 Multiplier (proporcionar módulo adecuado)
	fp_multiplier #(.EXP_WIDTH(5), .MAN_WIDTH(10), .BIAS(15)) fp16_mult (
		.a(a16),
		.b(b16),
		.result(fp16_mult_result)
	);

    always @(*) begin
        case (ALUControl)
            3'b000: Result = fp32_adder_result;
			//rotarlo a la izquierda 
            3'b101: Result = fp16_adder_result << 16;
            3'b100: Result = fp32_mult_result; 
            //rotarlo a la izquierda 
            3'b110: Result = fp16_mult_result << 16 ; 
            default: Result = 32'b0;
        endcase
    end

endmodule



module fp_adder #(
    parameter EXP_WIDTH = 8,          // Ancho del exponente
    parameter MAN_WIDTH = 23,         // Ancho de la mantisa
    parameter BIAS = 127              // Sesgo (bias)
)(
    input [EXP_WIDTH + MAN_WIDTH : 0] a,  // Primer operando
    input [EXP_WIDTH + MAN_WIDTH : 0] b,  // Segundo operando
    output [EXP_WIDTH + MAN_WIDTH : 0] result  // Resultado
);

    // Descomponer los números de punto flotante
    wire sign_a = a[EXP_WIDTH + MAN_WIDTH];
    wire [EXP_WIDTH-1:0] exp_a = a[EXP_WIDTH + MAN_WIDTH-1:MAN_WIDTH];
    wire [MAN_WIDTH:0] man_a = {1'b1, a[MAN_WIDTH-1:0]}; // Agregar el bit 1

    wire sign_b = b[EXP_WIDTH + MAN_WIDTH];
    wire [EXP_WIDTH-1:0] exp_b = b[EXP_WIDTH + MAN_WIDTH-1:MAN_WIDTH];
    wire [MAN_WIDTH:0] man_b = {1'b1, b[MAN_WIDTH-1:0]}; // Agregar el bit 1

    // Alinear las mantisas
    wire [EXP_WIDTH-1:0] exp_diff = (exp_a > exp_b) ? (exp_a - exp_b) : (exp_b - exp_a);
    wire [MAN_WIDTH:0] man_a_aligned = (exp_a > exp_b) ? man_a : (man_a >> exp_diff);
    wire [MAN_WIDTH:0] man_b_aligned = (exp_b > exp_a) ? man_b : (man_b >> exp_diff);
    wire [EXP_WIDTH-1:0] exp = (exp_a > exp_b) ? exp_a : exp_b;

    // Sumar o restar las mantisas (depende del signo)
    wire [MAN_WIDTH+1:0] sum = (sign_a == sign_b) ? (man_a_aligned + man_b_aligned) : 
                      ((man_a_aligned > man_b_aligned) ? (man_a_aligned - man_b_aligned) : 
                                                        (man_b_aligned - man_a_aligned));

    // Determinar el signo 
    wire result_sign = (sign_a == sign_b) ? sign_a : (man_a_aligned > man_b_aligned) ? sign_a : sign_b;

    // Normalización del resultado
    reg [MAN_WIDTH:0] normalized_man;
    reg [EXP_WIDTH-1:0] normalized_exp;
    always @(*) begin
        normalized_man = sum[MAN_WIDTH:0];
        normalized_exp = exp;
        if (sum[MAN_WIDTH+1]) begin
            // Si hay un acarreo, desplazar a la derecha
            normalized_man = sum[MAN_WIDTH:1];
            normalized_exp = exp + 1;
        end else begin
            // No hay acarreo, simplemente copiar los valores
            normalized_man = sum[MAN_WIDTH-1:0];
            normalized_exp = exp;
        end
    end

    // Resultado final
    assign result = {result_sign, normalized_exp, normalized_man[MAN_WIDTH-1:0]};

endmodule


module mux2 (
	d0,
	d1,
	s,
	y
);
	parameter WIDTH = 8;
	input wire [WIDTH - 1:0] d0;
	input wire [WIDTH - 1:0] d1;
	input wire s;
	output wire [WIDTH - 1:0] y;
	assign y = (s ? d1 : d0);
endmodule



module fp_multiplier #(
    parameter EXP_WIDTH = 8,          // Ancho del exponente
    parameter MAN_WIDTH = 23,         // Ancho de la mantisa
    parameter BIAS = 127              // Sesgo (bias)
)(
    input [EXP_WIDTH + MAN_WIDTH : 0] a,  // Primer operando
    input [EXP_WIDTH + MAN_WIDTH : 0] b,  // Segundo operando
    output [EXP_WIDTH + MAN_WIDTH : 0] result  // Resultado
);

    // Descomponer los números de punto flotante
    wire sign_a = a[EXP_WIDTH + MAN_WIDTH];
    wire [EXP_WIDTH-1:0] exp_a = a[EXP_WIDTH + MAN_WIDTH-1:MAN_WIDTH];
    wire [MAN_WIDTH:0] man_a = {1'b1, a[MAN_WIDTH-1:0]}; // Agregar el bit 1

    wire sign_b = b[EXP_WIDTH + MAN_WIDTH];
    wire [EXP_WIDTH-1:0] exp_b = b[EXP_WIDTH + MAN_WIDTH-1:MAN_WIDTH];
    wire [MAN_WIDTH:0] man_b = {1'b1, b[MAN_WIDTH-1:0]}; // Agregar el bit 1

    // Multiplicar las mantisas
    wire [2*MAN_WIDTH+1:0] man_product = man_a * man_b;

    // Sumar los exponentes y restar el bias
    wire [EXP_WIDTH+1:0] exp_sum = exp_a + exp_b - BIAS;

    // Determinar el signo del resultado
    wire result_sign = sign_a ^ sign_b;

    // Normalización del resultado
    reg [MAN_WIDTH:0] normalized_man;
    reg [EXP_WIDTH-1:0] normalized_exp;
    always @(*) begin
        if (man_product[2*MAN_WIDTH+1]) begin
            // Caso de overflow en la mantisa, desplazar a la derecha
            normalized_man = man_product[2*MAN_WIDTH+1:MAN_WIDTH+1];
            normalized_exp = exp_sum + 1;
        end else begin
            // Caso normal
            normalized_man = man_product[2*MAN_WIDTH:MAN_WIDTH];
            normalized_exp = exp_sum[EXP_WIDTH-1:0];
        end
    end

    // Manejo de casos especiales: Overflow y Underflow
    wire overflow = (normalized_exp >= (1 << EXP_WIDTH) - 1);
    wire underflow = (normalized_exp == 0);

    // Resultado final con manejo de casos especiales
    assign result = (overflow) ? {result_sign, {EXP_WIDTH{1'b1}}, {MAN_WIDTH{1'b0}}} : // Infinito
                     (underflow) ? {result_sign, {EXP_WIDTH{1'b0}}, {MAN_WIDTH{1'b0}}} : // Cero
                     {result_sign, normalized_exp, normalized_man[MAN_WIDTH-1:0]};

endmodule

module VRegFile (
	clk,
	vector_we,
	vector_wev,
	vector_ra1,
	vector_ra2,
	vector_wa,
	vector_wd,
	vector_wv,
	vector_rd1,
	vector_rd2,
	// Read register for display
	sel,
	vReg
);

		input wire clk;
    input wire vector_we;              // Señal de habilitación de escritura
		input wire vector_wev;						 // Señal que determina si escribir todo el vector o solo un valor.
    input wire [3:0] vector_ra1;       // Índice del primer registro a leer
    input wire [3:0] vector_ra2;       // Índice del segundo registro a leer
    input wire [3:0] vector_wa;        // Índice del registro a escribir
    input wire [159:0] vector_wd;      // Datos a escribir (5 elementos de 32 bits)
		input wire [31:0] vector_wv;			 // Valor a escribir en el vector.
    output wire [159:0] vector_rd1;    // Datos leídos del primer registro (5 elementos de 32 bits)
    output wire [159:0] vector_rd2;    // Datos leídos del segundo registro (5 elementos de 32 bits)
    // Definición del banco de registros vectoriales
    reg [31:0] vector_rf [15:0][4:0]; // 8 registros vectoriales, cada uno con 5 elementos de 32 bits

    // Escritura en registros vectoriales
    always @(posedge clk) begin
			if (vector_we) begin
				if (~vector_wev) begin
					vector_rf[vector_wa][0] <= vector_wd[31:0];
					vector_rf[vector_wa][1] <= vector_wd[63:32];
					vector_rf[vector_wa][2] <= vector_wd[95:64];
					vector_rf[vector_wa][3] <= vector_wd[127:96];
					vector_rf[vector_wa][4] <= vector_wd[159:128];
				end else begin
					vector_rf[vector_wa][vector_ra1] <= vector_wv;
				end
			end
    end

		// Lectura de registros vectoriales
		assign vector_rd1 = {vector_rf[vector_ra1][0], vector_rf[vector_ra1][1], vector_rf[vector_ra1][2], vector_rf[vector_ra1][3], vector_rf[vector_ra1][4]};
		assign vector_rd2 = {vector_rf[vector_ra2][0], vector_rf[vector_ra2][1], vector_rf[vector_ra2][2], vector_rf[vector_ra2][3], vector_rf[vector_ra2][4]};

		// Read register for display
		input wire [3:0] sel;
		output wire [159:0] vReg;
		assign vReg = {vector_rf[sel][0], vector_rf[sel][1], vector_rf[sel][2], vector_rf[sel][3], vector_rf[sel][4]};
endmodule
module Valu (
	Va,    
	Vb, 
	ALUControl,    
	IsFP,
	VResult,  // Vector de salida (5 elementos de 32 bits)
	VALUFlags
);
  
  input wire [159:0] Va;
  input wire [159:0] Vb;
  input wire [2:0] ALUControl;    
  input wire IsFP;
  output [159:0] VResult;  // Vector de salida (5 elementos de 32 bits)
  output [19:0] VALUFlags;  // Flags de la ALU para cada operación

  
  // Instancias de la ALU normal para cada par de elementos
	wire [31:0] VResult0, VResult1, VResult2, VResult3, VResult4;
	wire [3:0] VALUFlags0, VALUFlags1, VALUFlags2, VALUFlags3, VALUFlags4;

  alu alu0 (
    .a(Va[31:0]),
    .b(Vb[31:0]),
    .ALUControl(ALUControl), 
		.Result(VResult0),
		.ALUFlags(VALUFlags0),
	.IsFP(IsFP)
  );
  alu alu1 (
    .a(Va[63:32]),
    .b(Vb[63:32]),
    .ALUControl(ALUControl), 
		.Result(VResult1),
		.ALUFlags(VALUFlags1),
		.IsFP(IsFP)
  );
  alu alu2 (
    .a(Va[95:64]),
    .b(Vb[95:64]),
    .ALUControl(ALUControl), 
		.Result(VResult2),
		.ALUFlags(VALUFlags2),
		.IsFP(IsFP)
  );
  alu alu3 (
    .a(Va[127:96]),
    .b(Vb[127:96]),
    .ALUControl(ALUControl),
		.Result(VResult3),
		.ALUFlags(VALUFlags3),
		.IsFP(IsFP)
  );
  alu alu4 (
    .a(Va[159:128]),
    .b(Vb[159:128]),
    .ALUControl(ALUControl), 
		.Result(VResult4),
		.ALUFlags(VALUFlags4),
		.IsFP(IsFP)
  );

	assign VResult = {VResult0, VResult1, VResult2, VResult3, VResult4};
	assign VALUFlags = {VALUFlags0, VALUFlags1, VALUFlags2, VALUFlags3, VALUFlags4};
endmodule