0
ПЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8іЖ,
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
Х*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:*
dtype0
w
p_re_lu_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namep_re_lu_1/alpha
p
#p_re_lu_1/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_1/alpha*
_output_shapes	
:*
dtype0
y
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*
shared_namedense_7/kernel
r
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes
:		*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:	*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
Ю
8transformer_block/multi_head_self_attention/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*I
shared_name:8transformer_block/multi_head_self_attention/dense/kernel
Ч
Ltransformer_block/multi_head_self_attention/dense/kernel/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense/kernel* 
_output_shapes
:
ХХ*
dtype0
Х
6transformer_block/multi_head_self_attention/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*G
shared_name86transformer_block/multi_head_self_attention/dense/bias
О
Jtransformer_block/multi_head_self_attention/dense/bias/Read/ReadVariableOpReadVariableOp6transformer_block/multi_head_self_attention/dense/bias*
_output_shapes	
:Х*
dtype0
в
:transformer_block/multi_head_self_attention/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*K
shared_name<:transformer_block/multi_head_self_attention/dense_1/kernel
Ы
Ntransformer_block/multi_head_self_attention/dense_1/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_1/kernel* 
_output_shapes
:
ХХ*
dtype0
Щ
8transformer_block/multi_head_self_attention/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*I
shared_name:8transformer_block/multi_head_self_attention/dense_1/bias
Т
Ltransformer_block/multi_head_self_attention/dense_1/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_1/bias*
_output_shapes	
:Х*
dtype0
в
:transformer_block/multi_head_self_attention/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*K
shared_name<:transformer_block/multi_head_self_attention/dense_2/kernel
Ы
Ntransformer_block/multi_head_self_attention/dense_2/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_2/kernel* 
_output_shapes
:
ХХ*
dtype0
Щ
8transformer_block/multi_head_self_attention/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*I
shared_name:8transformer_block/multi_head_self_attention/dense_2/bias
Т
Ltransformer_block/multi_head_self_attention/dense_2/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_2/bias*
_output_shapes	
:Х*
dtype0
в
:transformer_block/multi_head_self_attention/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*K
shared_name<:transformer_block/multi_head_self_attention/dense_3/kernel
Ы
Ntransformer_block/multi_head_self_attention/dense_3/kernel/Read/ReadVariableOpReadVariableOp:transformer_block/multi_head_self_attention/dense_3/kernel* 
_output_shapes
:
ХХ*
dtype0
Щ
8transformer_block/multi_head_self_attention/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*I
shared_name:8transformer_block/multi_head_self_attention/dense_3/bias
Т
Ltransformer_block/multi_head_self_attention/dense_3/bias/Read/ReadVariableOpReadVariableOp8transformer_block/multi_head_self_attention/dense_3/bias*
_output_shapes	
:Х*
dtype0
Д
+transformer_block/sequential/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*<
shared_name-+transformer_block/sequential/dense_4/kernel
­
?transformer_block/sequential/dense_4/kernel/Read/ReadVariableOpReadVariableOp+transformer_block/sequential/dense_4/kernel* 
_output_shapes
:
Х*
dtype0
Ћ
)transformer_block/sequential/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)transformer_block/sequential/dense_4/bias
Є
=transformer_block/sequential/dense_4/bias/Read/ReadVariableOpReadVariableOp)transformer_block/sequential/dense_4/bias*
_output_shapes	
:*
dtype0
С
2transformer_block/sequential/dense_4/p_re_lu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*C
shared_name42transformer_block/sequential/dense_4/p_re_lu/alpha
К
Ftransformer_block/sequential/dense_4/p_re_lu/alpha/Read/ReadVariableOpReadVariableOp2transformer_block/sequential/dense_4/p_re_lu/alpha*
_output_shapes
:	2*
dtype0
Д
+transformer_block/sequential/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*<
shared_name-+transformer_block/sequential/dense_5/kernel
­
?transformer_block/sequential/dense_5/kernel/Read/ReadVariableOpReadVariableOp+transformer_block/sequential/dense_5/kernel* 
_output_shapes
:
Х*
dtype0
Ћ
)transformer_block/sequential/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*:
shared_name+)transformer_block/sequential/dense_5/bias
Є
=transformer_block/sequential/dense_5/bias/Read/ReadVariableOpReadVariableOp)transformer_block/sequential/dense_5/bias*
_output_shapes	
:Х*
dtype0
Џ
+transformer_block/layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*<
shared_name-+transformer_block/layer_normalization/gamma
Ј
?transformer_block/layer_normalization/gamma/Read/ReadVariableOpReadVariableOp+transformer_block/layer_normalization/gamma*
_output_shapes	
:Х*
dtype0
­
*transformer_block/layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*;
shared_name,*transformer_block/layer_normalization/beta
І
>transformer_block/layer_normalization/beta/Read/ReadVariableOpReadVariableOp*transformer_block/layer_normalization/beta*
_output_shapes	
:Х*
dtype0
Г
-transformer_block/layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*>
shared_name/-transformer_block/layer_normalization_1/gamma
Ќ
Atransformer_block/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOp-transformer_block/layer_normalization_1/gamma*
_output_shapes	
:Х*
dtype0
Б
,transformer_block/layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*=
shared_name.,transformer_block/layer_normalization_1/beta
Њ
@transformer_block/layer_normalization_1/beta/Read/ReadVariableOpReadVariableOp,transformer_block/layer_normalization_1/beta*
_output_shapes	
:Х*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
Х*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:*
dtype0

Adam/p_re_lu_1/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/p_re_lu_1/alpha/m
~
*Adam/p_re_lu_1/alpha/m/Read/ReadVariableOpReadVariableOpAdam/p_re_lu_1/alpha/m*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes
:		*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:	*
dtype0
м
?Adam/transformer_block/multi_head_self_attention/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense/kernel/m
е
SAdam/transformer_block/multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense/kernel/m* 
_output_shapes
:
ХХ*
dtype0
г
=Adam/transformer_block/multi_head_self_attention/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*N
shared_name?=Adam/transformer_block/multi_head_self_attention/dense/bias/m
Ь
QAdam/transformer_block/multi_head_self_attention/dense/bias/m/Read/ReadVariableOpReadVariableOp=Adam/transformer_block/multi_head_self_attention/dense/bias/m*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m
й
UAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m
а
SAdam/transformer_block/multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m
й
UAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m
а
SAdam/transformer_block/multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m
й
UAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m
а
SAdam/transformer_block/multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m*
_output_shapes	
:Х*
dtype0
Т
2Adam/transformer_block/sequential/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*C
shared_name42Adam/transformer_block/sequential/dense_4/kernel/m
Л
FAdam/transformer_block/sequential/dense_4/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_4/kernel/m* 
_output_shapes
:
Х*
dtype0
Й
0Adam/transformer_block/sequential/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/transformer_block/sequential/dense_4/bias/m
В
DAdam/transformer_block/sequential/dense_4/bias/m/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_4/bias/m*
_output_shapes	
:*
dtype0
Я
9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*J
shared_name;9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/m
Ш
MAdam/transformer_block/sequential/dense_4/p_re_lu/alpha/m/Read/ReadVariableOpReadVariableOp9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/m*
_output_shapes
:	2*
dtype0
Т
2Adam/transformer_block/sequential/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*C
shared_name42Adam/transformer_block/sequential/dense_5/kernel/m
Л
FAdam/transformer_block/sequential/dense_5/kernel/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_5/kernel/m* 
_output_shapes
:
Х*
dtype0
Й
0Adam/transformer_block/sequential/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*A
shared_name20Adam/transformer_block/sequential/dense_5/bias/m
В
DAdam/transformer_block/sequential/dense_5/bias/m/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_5/bias/m*
_output_shapes	
:Х*
dtype0
Н
2Adam/transformer_block/layer_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*C
shared_name42Adam/transformer_block/layer_normalization/gamma/m
Ж
FAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/m*
_output_shapes	
:Х*
dtype0
Л
1Adam/transformer_block/layer_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*B
shared_name31Adam/transformer_block/layer_normalization/beta/m
Д
EAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/m*
_output_shapes	
:Х*
dtype0
С
4Adam/transformer_block/layer_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/m
К
HAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/m*
_output_shapes	
:Х*
dtype0
П
3Adam/transformer_block/layer_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/m
И
GAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/m*
_output_shapes	
:Х*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
Х*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:*
dtype0

Adam/p_re_lu_1/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/p_re_lu_1/alpha/v
~
*Adam/p_re_lu_1/alpha/v/Read/ReadVariableOpReadVariableOpAdam/p_re_lu_1/alpha/v*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes
:		*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:	*
dtype0
м
?Adam/transformer_block/multi_head_self_attention/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense/kernel/v
е
SAdam/transformer_block/multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense/kernel/v* 
_output_shapes
:
ХХ*
dtype0
г
=Adam/transformer_block/multi_head_self_attention/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*N
shared_name?=Adam/transformer_block/multi_head_self_attention/dense/bias/v
Ь
QAdam/transformer_block/multi_head_self_attention/dense/bias/v/Read/ReadVariableOpReadVariableOp=Adam/transformer_block/multi_head_self_attention/dense/bias/v*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v
й
UAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v
а
SAdam/transformer_block/multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v
й
UAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v
а
SAdam/transformer_block/multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v*
_output_shapes	
:Х*
dtype0
р
AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ХХ*R
shared_nameCAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v
й
UAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v* 
_output_shapes
:
ХХ*
dtype0
з
?Adam/transformer_block/multi_head_self_attention/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*P
shared_nameA?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v
а
SAdam/transformer_block/multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOpReadVariableOp?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v*
_output_shapes	
:Х*
dtype0
Т
2Adam/transformer_block/sequential/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*C
shared_name42Adam/transformer_block/sequential/dense_4/kernel/v
Л
FAdam/transformer_block/sequential/dense_4/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_4/kernel/v* 
_output_shapes
:
Х*
dtype0
Й
0Adam/transformer_block/sequential/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adam/transformer_block/sequential/dense_4/bias/v
В
DAdam/transformer_block/sequential/dense_4/bias/v/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_4/bias/v*
_output_shapes	
:*
dtype0
Я
9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	2*J
shared_name;9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/v
Ш
MAdam/transformer_block/sequential/dense_4/p_re_lu/alpha/v/Read/ReadVariableOpReadVariableOp9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/v*
_output_shapes
:	2*
dtype0
Т
2Adam/transformer_block/sequential/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Х*C
shared_name42Adam/transformer_block/sequential/dense_5/kernel/v
Л
FAdam/transformer_block/sequential/dense_5/kernel/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/sequential/dense_5/kernel/v* 
_output_shapes
:
Х*
dtype0
Й
0Adam/transformer_block/sequential/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*A
shared_name20Adam/transformer_block/sequential/dense_5/bias/v
В
DAdam/transformer_block/sequential/dense_5/bias/v/Read/ReadVariableOpReadVariableOp0Adam/transformer_block/sequential/dense_5/bias/v*
_output_shapes	
:Х*
dtype0
Н
2Adam/transformer_block/layer_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*C
shared_name42Adam/transformer_block/layer_normalization/gamma/v
Ж
FAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpReadVariableOp2Adam/transformer_block/layer_normalization/gamma/v*
_output_shapes	
:Х*
dtype0
Л
1Adam/transformer_block/layer_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*B
shared_name31Adam/transformer_block/layer_normalization/beta/v
Д
EAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpReadVariableOp1Adam/transformer_block/layer_normalization/beta/v*
_output_shapes	
:Х*
dtype0
С
4Adam/transformer_block/layer_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*E
shared_name64Adam/transformer_block/layer_normalization_1/gamma/v
К
HAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/transformer_block/layer_normalization_1/gamma/v*
_output_shapes	
:Х*
dtype0
П
3Adam/transformer_block/layer_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Х*D
shared_name53Adam/transformer_block/layer_normalization_1/beta/v
И
GAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp3Adam/transformer_block/layer_normalization_1/beta/v*
_output_shapes	
:Х*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю
valueУBП BЗ
С
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
 
att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
 	keras_api
h

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
]
	'alpha
(	variables
)regularization_losses
*trainable_variables
+	keras_api
R
,	variables
-regularization_losses
.trainable_variables
/	keras_api
h

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
ј
6iter

7beta_1

8beta_2
	9decay
:learning_rate!mџ"m'm0m1m;m<m=m>m?m@mAmBmCmDmEmFmGmHmImJmKm!v"v'v0v1v;v<v=v>v?v@vAv BvЁCvЂDvЃEvЄFvЅGvІHvЇIvЈJvЉKvЊ
І
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
!17
"18
'19
020
121
 
І
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
!17
"18
'19
020
121
­
Llayer_regularization_losses
Mlayer_metrics

Nlayers

	variables
regularization_losses
Ometrics
Pnon_trainable_variables
trainable_variables
 

Qquery_dense
R	key_dense
Svalue_dense
Tcombine_heads
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
 
Ylayer_with_weights-0
Ylayer-0
Zlayer_with_weights-1
Zlayer-1
[	variables
\regularization_losses
]trainable_variables
^	keras_api
q
_axis
	Hgamma
Ibeta
`	variables
aregularization_losses
btrainable_variables
c	keras_api
q
daxis
	Jgamma
Kbeta
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
R
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
R
m	variables
nregularization_losses
otrainable_variables
p	keras_api
~
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
 
~
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
­
qlayer_regularization_losses
rlayer_metrics

slayers
	variables
regularization_losses
tmetrics
unon_trainable_variables
trainable_variables
 
 
 
­
vlayer_regularization_losses
wlayer_metrics

xlayers
	variables
regularization_losses
ymetrics
znon_trainable_variables
trainable_variables
 
 
 
­
{layer_regularization_losses
|layer_metrics

}layers
	variables
regularization_losses
~metrics
non_trainable_variables
trainable_variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1
 

!0
"1
В
 layer_regularization_losses
layer_metrics
layers
#	variables
$regularization_losses
metrics
non_trainable_variables
%trainable_variables
ZX
VARIABLE_VALUEp_re_lu_1/alpha5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUE

'0
 

'0
В
 layer_regularization_losses
layer_metrics
layers
(	variables
)regularization_losses
metrics
non_trainable_variables
*trainable_variables
 
 
 
В
 layer_regularization_losses
layer_metrics
layers
,	variables
-regularization_losses
metrics
non_trainable_variables
.trainable_variables
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11
 

00
11
В
 layer_regularization_losses
layer_metrics
layers
2	variables
3regularization_losses
metrics
non_trainable_variables
4trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE6transformer_block/multi_head_self_attention/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE:transformer_block/multi_head_self_attention/dense_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE8transformer_block/multi_head_self_attention/dense_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE+transformer_block/sequential/dense_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE)transformer_block/sequential/dense_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2transformer_block/sequential/dense_4/p_re_lu/alpha'variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+transformer_block/sequential/dense_5/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)transformer_block/sequential/dense_5/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+transformer_block/layer_normalization/gamma'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*transformer_block/layer_normalization/beta'variables/14/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-transformer_block/layer_normalization_1/gamma'variables/15/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE,transformer_block/layer_normalization_1/beta'variables/16/.ATTRIBUTES/VARIABLE_VALUE
 
 
8
0
1
2
3
4
5
6
7

0
1
 
l

;kernel
<bias
	variables
regularization_losses
trainable_variables
	keras_api
l

=kernel
>bias
	variables
regularization_losses
trainable_variables
	keras_api
l

?kernel
@bias
	variables
regularization_losses
 trainable_variables
Ё	keras_api
l

Akernel
Bbias
Ђ	variables
Ѓregularization_losses
Єtrainable_variables
Ѕ	keras_api
8
;0
<1
=2
>3
?4
@5
A6
B7
 
8
;0
<1
=2
>3
?4
@5
A6
B7
В
 Іlayer_regularization_losses
Їlayer_metrics
Јlayers
U	variables
Vregularization_losses
Љmetrics
Њnon_trainable_variables
Wtrainable_variables

Ћ
activation
Ќ_inbound_nodes

Ckernel
Dbias
­	variables
Ўregularization_losses
Џtrainable_variables
А	keras_api

Б_inbound_nodes

Fkernel
Gbias
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
#
C0
D1
E2
F3
G4
 
#
C0
D1
E2
F3
G4
В
 Жlayer_regularization_losses
Зlayer_metrics
Иlayers
[	variables
\regularization_losses
Йmetrics
Кnon_trainable_variables
]trainable_variables
 

H0
I1
 

H0
I1
В
 Лlayer_regularization_losses
Мlayer_metrics
Нlayers
`	variables
aregularization_losses
Оmetrics
Пnon_trainable_variables
btrainable_variables
 

J0
K1
 

J0
K1
В
 Рlayer_regularization_losses
Сlayer_metrics
Тlayers
e	variables
fregularization_losses
Уmetrics
Фnon_trainable_variables
gtrainable_variables
 
 
 
В
 Хlayer_regularization_losses
Цlayer_metrics
Чlayers
i	variables
jregularization_losses
Шmetrics
Щnon_trainable_variables
ktrainable_variables
 
 
 
В
 Ъlayer_regularization_losses
Ыlayer_metrics
Ьlayers
m	variables
nregularization_losses
Эmetrics
Юnon_trainable_variables
otrainable_variables
 
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

Яtotal

аcount
б	variables
в	keras_api
I

гtotal

дcount
е
_fn_kwargs
ж	variables
з	keras_api

;0
<1
 

;0
<1
Е
 иlayer_regularization_losses
йlayer_metrics
кlayers
	variables
regularization_losses
лmetrics
мnon_trainable_variables
trainable_variables

=0
>1
 

=0
>1
Е
 нlayer_regularization_losses
оlayer_metrics
пlayers
	variables
regularization_losses
рmetrics
сnon_trainable_variables
trainable_variables

?0
@1
 

?0
@1
Е
 тlayer_regularization_losses
уlayer_metrics
фlayers
	variables
regularization_losses
хmetrics
цnon_trainable_variables
 trainable_variables

A0
B1
 

A0
B1
Е
 чlayer_regularization_losses
шlayer_metrics
щlayers
Ђ	variables
Ѓregularization_losses
ъmetrics
ыnon_trainable_variables
Єtrainable_variables
 
 

Q0
R1
S2
T3
 
 
a
	Ealpha
ь	variables
эregularization_losses
юtrainable_variables
я	keras_api
 

C0
D1
E2
 

C0
D1
E2
Е
 №layer_regularization_losses
ёlayer_metrics
ђlayers
­	variables
Ўregularization_losses
ѓmetrics
єnon_trainable_variables
Џtrainable_variables
 

F0
G1
 

F0
G1
Е
 ѕlayer_regularization_losses
іlayer_metrics
їlayers
В	variables
Гregularization_losses
јmetrics
љnon_trainable_variables
Дtrainable_variables
 
 

Y0
Z1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Я0
а1

б	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

г0
д1

ж	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

E0
 

E0
Е
 њlayer_regularization_losses
ћlayer_metrics
ќlayers
ь	variables
эregularization_losses
§metrics
ўnon_trainable_variables
юtrainable_variables
 
 

Ћ0
 
 
 
 
 
 
 
 
 
 
 
 
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/p_re_lu_1/alpha/mQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/transformer_block/multi_head_self_attention/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_3/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/sequential/dense_4/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/transformer_block/sequential/dense_4/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/sequential/dense_5/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/transformer_block/sequential/dense_5/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/p_re_lu_1/alpha/vQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE=Adam/transformer_block/multi_head_self_attention/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE?Adam/transformer_block/multi_head_self_attention/dense_3/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/sequential/dense_4/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/transformer_block/sequential/dense_4/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/sequential/dense_5/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE0Adam/transformer_block/sequential/dense_5/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE2Adam/transformer_block/layer_normalization/gamma/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE1Adam/transformer_block/layer_normalization/beta/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE4Adam/transformer_block/layer_normalization_1/gamma/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE3Adam/transformer_block/layer_normalization_1/beta/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*,
_output_shapes
:џџџџџџџџџ2Х*
dtype0*!
shape:џџџџџџџџџ2Х


StatefulPartitionedCallStatefulPartitionedCallserving_default_input_18transformer_block/multi_head_self_attention/dense/kernel6transformer_block/multi_head_self_attention/dense/bias:transformer_block/multi_head_self_attention/dense_1/kernel8transformer_block/multi_head_self_attention/dense_1/bias:transformer_block/multi_head_self_attention/dense_2/kernel8transformer_block/multi_head_self_attention/dense_2/bias:transformer_block/multi_head_self_attention/dense_3/kernel8transformer_block/multi_head_self_attention/dense_3/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta+transformer_block/sequential/dense_4/kernel)transformer_block/sequential/dense_4/bias2transformer_block/sequential/dense_4/p_re_lu/alpha+transformer_block/sequential/dense_5/kernel)transformer_block/sequential/dense_5/bias-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betadense_6/kerneldense_6/biasp_re_lu_1/alphadense_7/kerneldense_7/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_65504
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
(
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp#p_re_lu_1/alpha/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense/kernel/Read/ReadVariableOpJtransformer_block/multi_head_self_attention/dense/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_1/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_1/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_2/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_2/bias/Read/ReadVariableOpNtransformer_block/multi_head_self_attention/dense_3/kernel/Read/ReadVariableOpLtransformer_block/multi_head_self_attention/dense_3/bias/Read/ReadVariableOp?transformer_block/sequential/dense_4/kernel/Read/ReadVariableOp=transformer_block/sequential/dense_4/bias/Read/ReadVariableOpFtransformer_block/sequential/dense_4/p_re_lu/alpha/Read/ReadVariableOp?transformer_block/sequential/dense_5/kernel/Read/ReadVariableOp=transformer_block/sequential/dense_5/bias/Read/ReadVariableOp?transformer_block/layer_normalization/gamma/Read/ReadVariableOp>transformer_block/layer_normalization/beta/Read/ReadVariableOpAtransformer_block/layer_normalization_1/gamma/Read/ReadVariableOp@transformer_block/layer_normalization_1/beta/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp*Adam/p_re_lu_1/alpha/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense/kernel/m/Read/ReadVariableOpQAdam/transformer_block/multi_head_self_attention/dense/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_1/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_2/bias/m/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_3/bias/m/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_4/kernel/m/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_4/bias/m/Read/ReadVariableOpMAdam/transformer_block/sequential/dense_4/p_re_lu/alpha/m/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_5/kernel/m/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_5/bias/m/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/m/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/m/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/m/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/m/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp*Adam/p_re_lu_1/alpha/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense/kernel/v/Read/ReadVariableOpQAdam/transformer_block/multi_head_self_attention/dense/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_1/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_2/bias/v/Read/ReadVariableOpUAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v/Read/ReadVariableOpSAdam/transformer_block/multi_head_self_attention/dense_3/bias/v/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_4/kernel/v/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_4/bias/v/Read/ReadVariableOpMAdam/transformer_block/sequential/dense_4/p_re_lu/alpha/v/Read/ReadVariableOpFAdam/transformer_block/sequential/dense_5/kernel/v/Read/ReadVariableOpDAdam/transformer_block/sequential/dense_5/bias/v/Read/ReadVariableOpFAdam/transformer_block/layer_normalization/gamma/v/Read/ReadVariableOpEAdam/transformer_block/layer_normalization/beta/v/Read/ReadVariableOpHAdam/transformer_block/layer_normalization_1/gamma/v/Read/ReadVariableOpGAdam/transformer_block/layer_normalization_1/beta/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_67585
Њ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasp_re_lu_1/alphadense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate8transformer_block/multi_head_self_attention/dense/kernel6transformer_block/multi_head_self_attention/dense/bias:transformer_block/multi_head_self_attention/dense_1/kernel8transformer_block/multi_head_self_attention/dense_1/bias:transformer_block/multi_head_self_attention/dense_2/kernel8transformer_block/multi_head_self_attention/dense_2/bias:transformer_block/multi_head_self_attention/dense_3/kernel8transformer_block/multi_head_self_attention/dense_3/bias+transformer_block/sequential/dense_4/kernel)transformer_block/sequential/dense_4/bias2transformer_block/sequential/dense_4/p_re_lu/alpha+transformer_block/sequential/dense_5/kernel)transformer_block/sequential/dense_5/bias+transformer_block/layer_normalization/gamma*transformer_block/layer_normalization/beta-transformer_block/layer_normalization_1/gamma,transformer_block/layer_normalization_1/betatotalcounttotal_1count_1Adam/dense_6/kernel/mAdam/dense_6/bias/mAdam/p_re_lu_1/alpha/mAdam/dense_7/kernel/mAdam/dense_7/bias/m?Adam/transformer_block/multi_head_self_attention/dense/kernel/m=Adam/transformer_block/multi_head_self_attention/dense/bias/mAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_1/bias/mAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_2/bias/mAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m2Adam/transformer_block/sequential/dense_4/kernel/m0Adam/transformer_block/sequential/dense_4/bias/m9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/m2Adam/transformer_block/sequential/dense_5/kernel/m0Adam/transformer_block/sequential/dense_5/bias/m2Adam/transformer_block/layer_normalization/gamma/m1Adam/transformer_block/layer_normalization/beta/m4Adam/transformer_block/layer_normalization_1/gamma/m3Adam/transformer_block/layer_normalization_1/beta/mAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/p_re_lu_1/alpha/vAdam/dense_7/kernel/vAdam/dense_7/bias/v?Adam/transformer_block/multi_head_self_attention/dense/kernel/v=Adam/transformer_block/multi_head_self_attention/dense/bias/vAAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_1/bias/vAAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_2/bias/vAAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v2Adam/transformer_block/sequential/dense_4/kernel/v0Adam/transformer_block/sequential/dense_4/bias/v9Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/v2Adam/transformer_block/sequential/dense_5/kernel/v0Adam/transformer_block/sequential/dense_5/bias/v2Adam/transformer_block/layer_normalization/gamma/v1Adam/transformer_block/layer_normalization/beta/v4Adam/transformer_block/layer_normalization_1/gamma/v3Adam/transformer_block/layer_normalization_1/beta/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_67820№)
Ђ
b
)__inference_dropout_3_layer_call_fn_67066

inputs
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а
o
)__inference_p_re_lu_1_layer_call_fn_64317

inputs
unknown
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_643092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_67010

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџХ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџХ:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs
г
Њ
B__inference_dense_6_layer_call_and_return_conditional_losses_67035

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџХ:::P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs
ЖЇ
Ъ,
__inference__traced_save_67585
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop.
*savev2_p_re_lu_1_alpha_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_kernel_read_readvariableopU
Qsavev2_transformer_block_multi_head_self_attention_dense_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_1_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_1_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_2_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_2_bias_read_readvariableopY
Usavev2_transformer_block_multi_head_self_attention_dense_3_kernel_read_readvariableopW
Ssavev2_transformer_block_multi_head_self_attention_dense_3_bias_read_readvariableopJ
Fsavev2_transformer_block_sequential_dense_4_kernel_read_readvariableopH
Dsavev2_transformer_block_sequential_dense_4_bias_read_readvariableopQ
Msavev2_transformer_block_sequential_dense_4_p_re_lu_alpha_read_readvariableopJ
Fsavev2_transformer_block_sequential_dense_5_kernel_read_readvariableopH
Dsavev2_transformer_block_sequential_dense_5_bias_read_readvariableopJ
Fsavev2_transformer_block_layer_normalization_gamma_read_readvariableopI
Esavev2_transformer_block_layer_normalization_beta_read_readvariableopL
Hsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopK
Gsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop5
1savev2_adam_p_re_lu_1_alpha_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_m_read_readvariableop\
Xsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_m_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_m_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_4_kernel_m_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_4_bias_m_read_readvariableopX
Tsavev2_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_m_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_5_kernel_m_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_5_bias_m_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop5
1savev2_adam_p_re_lu_1_alpha_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_v_read_readvariableop\
Xsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_v_read_readvariableop`
\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_v_read_readvariableop^
Zsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_4_kernel_v_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_4_bias_v_read_readvariableopX
Tsavev2_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_v_read_readvariableopQ
Msavev2_adam_transformer_block_sequential_dense_5_kernel_v_read_readvariableopO
Ksavev2_adam_transformer_block_sequential_dense_5_bias_v_read_readvariableopQ
Msavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopP
Lsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopS
Osavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopR
Nsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_8a99b6fa290c4026a3e11a6e79470dbe/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameє$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*$
valueќ#Bљ#LB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЃ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesИ+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop*savev2_p_re_lu_1_alpha_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_kernel_read_readvariableopQsavev2_transformer_block_multi_head_self_attention_dense_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_1_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_1_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_2_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_2_bias_read_readvariableopUsavev2_transformer_block_multi_head_self_attention_dense_3_kernel_read_readvariableopSsavev2_transformer_block_multi_head_self_attention_dense_3_bias_read_readvariableopFsavev2_transformer_block_sequential_dense_4_kernel_read_readvariableopDsavev2_transformer_block_sequential_dense_4_bias_read_readvariableopMsavev2_transformer_block_sequential_dense_4_p_re_lu_alpha_read_readvariableopFsavev2_transformer_block_sequential_dense_5_kernel_read_readvariableopDsavev2_transformer_block_sequential_dense_5_bias_read_readvariableopFsavev2_transformer_block_layer_normalization_gamma_read_readvariableopEsavev2_transformer_block_layer_normalization_beta_read_readvariableopHsavev2_transformer_block_layer_normalization_1_gamma_read_readvariableopGsavev2_transformer_block_layer_normalization_1_beta_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop1savev2_adam_p_re_lu_1_alpha_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_m_read_readvariableopXsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_m_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_m_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_m_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_4_kernel_m_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_4_bias_m_read_readvariableopTsavev2_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_m_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_5_kernel_m_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_5_bias_m_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_m_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_m_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_m_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop1savev2_adam_p_re_lu_1_alpha_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_kernel_v_read_readvariableopXsavev2_adam_transformer_block_multi_head_self_attention_dense_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_1_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_1_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_2_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_2_bias_v_read_readvariableop\savev2_adam_transformer_block_multi_head_self_attention_dense_3_kernel_v_read_readvariableopZsavev2_adam_transformer_block_multi_head_self_attention_dense_3_bias_v_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_4_kernel_v_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_4_bias_v_read_readvariableopTsavev2_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_v_read_readvariableopMsavev2_adam_transformer_block_sequential_dense_5_kernel_v_read_readvariableopKsavev2_adam_transformer_block_sequential_dense_5_bias_v_read_readvariableopMsavev2_adam_transformer_block_layer_normalization_gamma_v_read_readvariableopLsavev2_adam_transformer_block_layer_normalization_beta_v_read_readvariableopOsavev2_adam_transformer_block_layer_normalization_1_gamma_v_read_readvariableopNsavev2_adam_transformer_block_layer_normalization_1_beta_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ї
_input_shapesх
т: :
Х:::		:	: : : : : :
ХХ:Х:
ХХ:Х:
ХХ:Х:
ХХ:Х:
Х::	2:
Х:Х:Х:Х:Х:Х: : : : :
Х:::		:	:
ХХ:Х:
ХХ:Х:
ХХ:Х:
ХХ:Х:
Х::	2:
Х:Х:Х:Х:Х:Х:
Х:::		:	:
ХХ:Х:
ХХ:Х:
ХХ:Х:
ХХ:Х:
Х::	2:
Х:Х:Х:Х:Х:Х: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
Х:!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:		: 

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :&"
 
_output_shapes
:
ХХ:!

_output_shapes	
:Х:&"
 
_output_shapes
:
ХХ:!

_output_shapes	
:Х:&"
 
_output_shapes
:
ХХ:!

_output_shapes	
:Х:&"
 
_output_shapes
:
ХХ:!

_output_shapes	
:Х:&"
 
_output_shapes
:
Х:!

_output_shapes	
::%!

_output_shapes
:	2:&"
 
_output_shapes
:
Х:!

_output_shapes	
:Х:!

_output_shapes	
:Х:!

_output_shapes	
:Х:!

_output_shapes	
:Х:!

_output_shapes	
:Х:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :& "
 
_output_shapes
:
Х:!!

_output_shapes	
::!"

_output_shapes	
::%#!

_output_shapes
:		: $

_output_shapes
:	:&%"
 
_output_shapes
:
ХХ:!&

_output_shapes	
:Х:&'"
 
_output_shapes
:
ХХ:!(

_output_shapes	
:Х:&)"
 
_output_shapes
:
ХХ:!*

_output_shapes	
:Х:&+"
 
_output_shapes
:
ХХ:!,

_output_shapes	
:Х:&-"
 
_output_shapes
:
Х:!.

_output_shapes	
::%/!

_output_shapes
:	2:&0"
 
_output_shapes
:
Х:!1

_output_shapes	
:Х:!2

_output_shapes	
:Х:!3

_output_shapes	
:Х:!4

_output_shapes	
:Х:!5

_output_shapes	
:Х:&6"
 
_output_shapes
:
Х:!7

_output_shapes	
::!8

_output_shapes	
::%9!

_output_shapes
:		: :

_output_shapes
:	:&;"
 
_output_shapes
:
ХХ:!<

_output_shapes	
:Х:&="
 
_output_shapes
:
ХХ:!>

_output_shapes	
:Х:&?"
 
_output_shapes
:
ХХ:!@

_output_shapes	
:Х:&A"
 
_output_shapes
:
ХХ:!B

_output_shapes	
:Х:&C"
 
_output_shapes
:
Х:!D

_output_shapes	
::%E!

_output_shapes
:	2:&F"
 
_output_shapes
:
Х:!G

_output_shapes	
:Х:!H

_output_shapes	
:Х:!I

_output_shapes	
:Х:!J

_output_shapes	
:Х:!K

_output_shapes	
:Х:L

_output_shapes
: 
ш
Г
*__inference_sequential_layer_call_fn_64246
dense_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_642332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ2Х
'
_user_specified_namedense_4_input
ь
|
'__inference_dense_5_layer_call_fn_67337

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_641812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
В
Њ
B__inference_dense_7_layer_call_and_return_conditional_losses_65164

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
М
T
8__inference_global_average_pooling1d_layer_call_fn_66987

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_650572
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ2Х:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
Ђ
b
)__inference_dropout_2_layer_call_fn_67020

inputs
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџХ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs

E
)__inference_dropout_3_layer_call_fn_67071

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651402
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
фЯ
є4
!__inference__traced_restore_67820
file_prefix#
assignvariableop_dense_6_kernel#
assignvariableop_1_dense_6_bias&
"assignvariableop_2_p_re_lu_1_alpha%
!assignvariableop_3_dense_7_kernel#
assignvariableop_4_dense_7_bias 
assignvariableop_5_adam_iter"
assignvariableop_6_adam_beta_1"
assignvariableop_7_adam_beta_2!
assignvariableop_8_adam_decay)
%assignvariableop_9_adam_learning_rateP
Lassignvariableop_10_transformer_block_multi_head_self_attention_dense_kernelN
Jassignvariableop_11_transformer_block_multi_head_self_attention_dense_biasR
Nassignvariableop_12_transformer_block_multi_head_self_attention_dense_1_kernelP
Lassignvariableop_13_transformer_block_multi_head_self_attention_dense_1_biasR
Nassignvariableop_14_transformer_block_multi_head_self_attention_dense_2_kernelP
Lassignvariableop_15_transformer_block_multi_head_self_attention_dense_2_biasR
Nassignvariableop_16_transformer_block_multi_head_self_attention_dense_3_kernelP
Lassignvariableop_17_transformer_block_multi_head_self_attention_dense_3_biasC
?assignvariableop_18_transformer_block_sequential_dense_4_kernelA
=assignvariableop_19_transformer_block_sequential_dense_4_biasJ
Fassignvariableop_20_transformer_block_sequential_dense_4_p_re_lu_alphaC
?assignvariableop_21_transformer_block_sequential_dense_5_kernelA
=assignvariableop_22_transformer_block_sequential_dense_5_biasC
?assignvariableop_23_transformer_block_layer_normalization_gammaB
>assignvariableop_24_transformer_block_layer_normalization_betaE
Aassignvariableop_25_transformer_block_layer_normalization_1_gammaD
@assignvariableop_26_transformer_block_layer_normalization_1_beta
assignvariableop_27_total
assignvariableop_28_count
assignvariableop_29_total_1
assignvariableop_30_count_1-
)assignvariableop_31_adam_dense_6_kernel_m+
'assignvariableop_32_adam_dense_6_bias_m.
*assignvariableop_33_adam_p_re_lu_1_alpha_m-
)assignvariableop_34_adam_dense_7_kernel_m+
'assignvariableop_35_adam_dense_7_bias_mW
Sassignvariableop_36_adam_transformer_block_multi_head_self_attention_dense_kernel_mU
Qassignvariableop_37_adam_transformer_block_multi_head_self_attention_dense_bias_mY
Uassignvariableop_38_adam_transformer_block_multi_head_self_attention_dense_1_kernel_mW
Sassignvariableop_39_adam_transformer_block_multi_head_self_attention_dense_1_bias_mY
Uassignvariableop_40_adam_transformer_block_multi_head_self_attention_dense_2_kernel_mW
Sassignvariableop_41_adam_transformer_block_multi_head_self_attention_dense_2_bias_mY
Uassignvariableop_42_adam_transformer_block_multi_head_self_attention_dense_3_kernel_mW
Sassignvariableop_43_adam_transformer_block_multi_head_self_attention_dense_3_bias_mJ
Fassignvariableop_44_adam_transformer_block_sequential_dense_4_kernel_mH
Dassignvariableop_45_adam_transformer_block_sequential_dense_4_bias_mQ
Massignvariableop_46_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_mJ
Fassignvariableop_47_adam_transformer_block_sequential_dense_5_kernel_mH
Dassignvariableop_48_adam_transformer_block_sequential_dense_5_bias_mJ
Fassignvariableop_49_adam_transformer_block_layer_normalization_gamma_mI
Eassignvariableop_50_adam_transformer_block_layer_normalization_beta_mL
Hassignvariableop_51_adam_transformer_block_layer_normalization_1_gamma_mK
Gassignvariableop_52_adam_transformer_block_layer_normalization_1_beta_m-
)assignvariableop_53_adam_dense_6_kernel_v+
'assignvariableop_54_adam_dense_6_bias_v.
*assignvariableop_55_adam_p_re_lu_1_alpha_v-
)assignvariableop_56_adam_dense_7_kernel_v+
'assignvariableop_57_adam_dense_7_bias_vW
Sassignvariableop_58_adam_transformer_block_multi_head_self_attention_dense_kernel_vU
Qassignvariableop_59_adam_transformer_block_multi_head_self_attention_dense_bias_vY
Uassignvariableop_60_adam_transformer_block_multi_head_self_attention_dense_1_kernel_vW
Sassignvariableop_61_adam_transformer_block_multi_head_self_attention_dense_1_bias_vY
Uassignvariableop_62_adam_transformer_block_multi_head_self_attention_dense_2_kernel_vW
Sassignvariableop_63_adam_transformer_block_multi_head_self_attention_dense_2_bias_vY
Uassignvariableop_64_adam_transformer_block_multi_head_self_attention_dense_3_kernel_vW
Sassignvariableop_65_adam_transformer_block_multi_head_self_attention_dense_3_bias_vJ
Fassignvariableop_66_adam_transformer_block_sequential_dense_4_kernel_vH
Dassignvariableop_67_adam_transformer_block_sequential_dense_4_bias_vQ
Massignvariableop_68_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_vJ
Fassignvariableop_69_adam_transformer_block_sequential_dense_5_kernel_vH
Dassignvariableop_70_adam_transformer_block_sequential_dense_5_bias_vJ
Fassignvariableop_71_adam_transformer_block_layer_normalization_gamma_vI
Eassignvariableop_72_adam_transformer_block_layer_normalization_beta_vL
Hassignvariableop_73_adam_transformer_block_layer_normalization_1_gamma_vK
Gassignvariableop_74_adam_transformer_block_layer_normalization_1_beta_v
identity_76ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_8ЂAssignVariableOp_9њ$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*$
valueќ#Bљ#LB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/alpha/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЉ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
valueЃB LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЊ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ц
_output_shapesГ
А::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Є
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_p_re_lu_1_alphaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_7_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Є
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_7_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5Ё
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ѓ
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ђ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Њ
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10д
AssignVariableOp_10AssignVariableOpLassignvariableop_10_transformer_block_multi_head_self_attention_dense_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11в
AssignVariableOp_11AssignVariableOpJassignvariableop_11_transformer_block_multi_head_self_attention_dense_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ж
AssignVariableOp_12AssignVariableOpNassignvariableop_12_transformer_block_multi_head_self_attention_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13д
AssignVariableOp_13AssignVariableOpLassignvariableop_13_transformer_block_multi_head_self_attention_dense_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ж
AssignVariableOp_14AssignVariableOpNassignvariableop_14_transformer_block_multi_head_self_attention_dense_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15д
AssignVariableOp_15AssignVariableOpLassignvariableop_15_transformer_block_multi_head_self_attention_dense_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ж
AssignVariableOp_16AssignVariableOpNassignvariableop_16_transformer_block_multi_head_self_attention_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpLassignvariableop_17_transformer_block_multi_head_self_attention_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ч
AssignVariableOp_18AssignVariableOp?assignvariableop_18_transformer_block_sequential_dense_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Х
AssignVariableOp_19AssignVariableOp=assignvariableop_19_transformer_block_sequential_dense_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ю
AssignVariableOp_20AssignVariableOpFassignvariableop_20_transformer_block_sequential_dense_4_p_re_lu_alphaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ч
AssignVariableOp_21AssignVariableOp?assignvariableop_21_transformer_block_sequential_dense_5_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Х
AssignVariableOp_22AssignVariableOp=assignvariableop_22_transformer_block_sequential_dense_5_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Ч
AssignVariableOp_23AssignVariableOp?assignvariableop_23_transformer_block_layer_normalization_gammaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ц
AssignVariableOp_24AssignVariableOp>assignvariableop_24_transformer_block_layer_normalization_betaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Щ
AssignVariableOp_25AssignVariableOpAassignvariableop_25_transformer_block_layer_normalization_1_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ш
AssignVariableOp_26AssignVariableOp@assignvariableop_26_transformer_block_layer_normalization_1_betaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ё
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ё
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ѓ
AssignVariableOp_29AssignVariableOpassignvariableop_29_total_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Ѓ
AssignVariableOp_30AssignVariableOpassignvariableop_30_count_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Б
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_6_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Џ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_6_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33В
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_p_re_lu_1_alpha_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Б
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_7_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Џ
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_7_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36л
AssignVariableOp_36AssignVariableOpSassignvariableop_36_adam_transformer_block_multi_head_self_attention_dense_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37й
AssignVariableOp_37AssignVariableOpQassignvariableop_37_adam_transformer_block_multi_head_self_attention_dense_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38н
AssignVariableOp_38AssignVariableOpUassignvariableop_38_adam_transformer_block_multi_head_self_attention_dense_1_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39л
AssignVariableOp_39AssignVariableOpSassignvariableop_39_adam_transformer_block_multi_head_self_attention_dense_1_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40н
AssignVariableOp_40AssignVariableOpUassignvariableop_40_adam_transformer_block_multi_head_self_attention_dense_2_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41л
AssignVariableOp_41AssignVariableOpSassignvariableop_41_adam_transformer_block_multi_head_self_attention_dense_2_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42н
AssignVariableOp_42AssignVariableOpUassignvariableop_42_adam_transformer_block_multi_head_self_attention_dense_3_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43л
AssignVariableOp_43AssignVariableOpSassignvariableop_43_adam_transformer_block_multi_head_self_attention_dense_3_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Ю
AssignVariableOp_44AssignVariableOpFassignvariableop_44_adam_transformer_block_sequential_dense_4_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ь
AssignVariableOp_45AssignVariableOpDassignvariableop_45_adam_transformer_block_sequential_dense_4_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46е
AssignVariableOp_46AssignVariableOpMassignvariableop_46_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Ю
AssignVariableOp_47AssignVariableOpFassignvariableop_47_adam_transformer_block_sequential_dense_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Ь
AssignVariableOp_48AssignVariableOpDassignvariableop_48_adam_transformer_block_sequential_dense_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ю
AssignVariableOp_49AssignVariableOpFassignvariableop_49_adam_transformer_block_layer_normalization_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Э
AssignVariableOp_50AssignVariableOpEassignvariableop_50_adam_transformer_block_layer_normalization_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51а
AssignVariableOp_51AssignVariableOpHassignvariableop_51_adam_transformer_block_layer_normalization_1_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52Я
AssignVariableOp_52AssignVariableOpGassignvariableop_52_adam_transformer_block_layer_normalization_1_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Б
AssignVariableOp_53AssignVariableOp)assignvariableop_53_adam_dense_6_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Џ
AssignVariableOp_54AssignVariableOp'assignvariableop_54_adam_dense_6_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55В
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_p_re_lu_1_alpha_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Б
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_7_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Џ
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_dense_7_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58л
AssignVariableOp_58AssignVariableOpSassignvariableop_58_adam_transformer_block_multi_head_self_attention_dense_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59й
AssignVariableOp_59AssignVariableOpQassignvariableop_59_adam_transformer_block_multi_head_self_attention_dense_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60н
AssignVariableOp_60AssignVariableOpUassignvariableop_60_adam_transformer_block_multi_head_self_attention_dense_1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61л
AssignVariableOp_61AssignVariableOpSassignvariableop_61_adam_transformer_block_multi_head_self_attention_dense_1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62н
AssignVariableOp_62AssignVariableOpUassignvariableop_62_adam_transformer_block_multi_head_self_attention_dense_2_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63л
AssignVariableOp_63AssignVariableOpSassignvariableop_63_adam_transformer_block_multi_head_self_attention_dense_2_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64н
AssignVariableOp_64AssignVariableOpUassignvariableop_64_adam_transformer_block_multi_head_self_attention_dense_3_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65л
AssignVariableOp_65AssignVariableOpSassignvariableop_65_adam_transformer_block_multi_head_self_attention_dense_3_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Ю
AssignVariableOp_66AssignVariableOpFassignvariableop_66_adam_transformer_block_sequential_dense_4_kernel_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ь
AssignVariableOp_67AssignVariableOpDassignvariableop_67_adam_transformer_block_sequential_dense_4_bias_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68е
AssignVariableOp_68AssignVariableOpMassignvariableop_68_adam_transformer_block_sequential_dense_4_p_re_lu_alpha_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ю
AssignVariableOp_69AssignVariableOpFassignvariableop_69_adam_transformer_block_sequential_dense_5_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ь
AssignVariableOp_70AssignVariableOpDassignvariableop_70_adam_transformer_block_sequential_dense_5_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71Ю
AssignVariableOp_71AssignVariableOpFassignvariableop_71_adam_transformer_block_layer_normalization_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72Э
AssignVariableOp_72AssignVariableOpEassignvariableop_72_adam_transformer_block_layer_normalization_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73а
AssignVariableOp_73AssignVariableOpHassignvariableop_73_adam_transformer_block_layer_normalization_1_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74Я
AssignVariableOp_74AssignVariableOpGassignvariableop_74_adam_transformer_block_layer_normalization_1_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_749
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpа
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_75У
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_76"#
identity_76Identity_76:output:0*У
_input_shapesБ
Ў: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
%
в
B__inference_dense_4_layer_call_and_return_conditional_losses_67287

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22	
BiasAddm
p_re_lu/ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Relu
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02
p_re_lu/ReadVariableOpk
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
p_re_lu/Negn
p_re_lu/Neg_1NegBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Neg_1r
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Relu_1
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/mul
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/addh
IdentityIdentityp_re_lu/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ2Х::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
г
Њ
B__inference_dense_6_layer_call_and_return_conditional_losses_65104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAdde
IdentityIdentityBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџХ:::P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs
хи
	
L__inference_transformer_block_layer_call_and_return_conditional_losses_64937

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource5
1layer_normalization_mul_3_readvariableop_resource3
/layer_normalization_add_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource6
2sequential_dense_4_p_re_lu_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource7
3layer_normalization_1_mul_3_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource
identityx
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:2!
multi_head_self_attention/ShapeЈ
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-multi_head_self_attention/strided_slice/stackЌ
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_1Ќ
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_2ў
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'multi_head_self_attention/strided_sliceј
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02:
8multi_head_self_attention/dense/Tensordot/ReadVariableOpЊ
.multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_self_attention/dense/Tensordot/axesБ
.multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_self_attention/dense/Tensordot/free
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/ShapeД
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/GatherV2/axisё
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/GatherV2И
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisї
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense/Tensordot/GatherV2_1Ќ
/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention/dense/Tensordot/Const
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_self_attention/dense/Tensordot/ProdА
1multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense/Tensordot/Const_1
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense/Tensordot/Prod_1А
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_self_attention/dense/Tensordot/concat/axisа
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_self_attention/dense/Tensordot/concat
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/stackё
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х25
3multi_head_self_attention/dense/Tensordot/transpose
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1multi_head_self_attention/dense/Tensordot/Reshape
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ22
0multi_head_self_attention/dense/Tensordot/MatMulБ
1multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х23
1multi_head_self_attention/dense/Tensordot/Const_2Д
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/concat_1/axisн
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/concat_1
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense/Tensordotэ
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype028
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2)
'multi_head_self_attention/dense/BiasAddў
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_1/Tensordot/axesЕ
0multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_1/Tensordot/free
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/ShapeИ
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/GatherV2М
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_1/Tensordot/Const
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_1/Tensordot/ProdД
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_1/Tensordot/Const_1
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_1/Tensordot/Prod_1Д
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_1/Tensordot/concat/axisк
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_1/Tensordot/concat
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/stackї
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_1/Tensordot/transposeЇ
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_1/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_1/Tensordot/MatMulЕ
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_1/Tensordot/Const_2И
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/concat_1
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_1/Tensordotѓ
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_1/BiasAddў
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_2/Tensordot/axesЕ
0multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_2/Tensordot/free
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/ShapeИ
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/GatherV2М
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_2/Tensordot/Const
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_2/Tensordot/ProdД
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_2/Tensordot/Const_1
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_2/Tensordot/Prod_1Д
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_2/Tensordot/concat/axisк
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_2/Tensordot/concat
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/stackї
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_2/Tensordot/transposeЇ
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_2/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_2/Tensordot/MatMulЕ
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_2/Tensordot/Const_2И
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/concat_1
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_2/Tensordotѓ
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_2/BiasAddЁ
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)multi_head_self_attention/Reshape/shape/1
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/2
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/3ж
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_self_attention/Reshape/shapeј
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Reshape­
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(multi_head_self_attention/transpose/permљ
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/transposeЅ
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_1/shape/1
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/2
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/3р
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_1/shape
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_1Б
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_1/perm
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_1Ѕ
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_2/shape/1
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/2
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/3р
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_2/shape
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_2Б
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_2/perm
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_2
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(2"
 multi_head_self_attention/MatMul
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:2#
!multi_head_self_attention/Shape_1Е
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ21
/multi_head_self_attention/strided_slice_1/stackА
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/strided_slice_1/stack_1А
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention/strided_slice_1/stack_2
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention/strided_slice_1Ќ
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
multi_head_self_attention/Cast
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 2 
multi_head_self_attention/Sqrtь
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/truedivФ
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Softmaxє
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2$
"multi_head_self_attention/MatMul_1Б
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_3/perm
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_3Ѕ
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_3/shape/1
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2-
+multi_head_self_attention/Reshape_3/shape/2Њ
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_3/shapeє
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2%
#multi_head_self_attention/Reshape_3ў
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_3/Tensordot/axesЕ
0multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_3/Tensordot/freeТ
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/ShapeИ
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/GatherV2М
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_3/Tensordot/Const
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_3/Tensordot/ProdД
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_3/Tensordot/Const_1
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_3/Tensordot/Prod_1Д
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_3/Tensordot/concat/axisк
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_3/Tensordot/concat
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/stackІ
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5multi_head_self_attention/dense_3/Tensordot/transposeЇ
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_3/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_3/Tensordot/MatMulЕ
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_3/Tensordot/Const_2И
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/concat_1Ђ
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2-
+multi_head_self_attention/dense_3/Tensordotѓ
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2+
)multi_head_self_attention/dense_3/BiasAddЄ
dropout/IdentityIdentity2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/Identitym
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
addm
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:2
layer_normalization/Shape
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack 
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1 
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2к
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/xЊ
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul 
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stackЄ
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1Є
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2ф
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1Љ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1 
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_2/stackЄ
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_1Є
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_2ф
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_2|
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_2/xВ
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_2
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shapeН
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization/Const
layer_normalization/Fill/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dimsЖ
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1
layer_normalization/Fill_1/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dimsО
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3Я
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2&
$layer_normalization/FusedBatchNormV3Ю
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/Reshape_1У
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization/mul_3/ReadVariableOpЮ
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/mul_3Н
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02(
&layer_normalization/add/ReadVariableOpС
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/addб
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_4/Tensordot/ReadVariableOp
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_4/Tensordot/axes
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_4/Tensordot/free
"sequential/dense_4/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/Shape
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/GatherV2/axisА
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/GatherV2
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_4/Tensordot/GatherV2_1/axisЖ
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_4/Tensordot/GatherV2_1
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_4/Tensordot/ConstЬ
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_4/Tensordot/Prod
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_4/Tensordot/Const_1д
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_4/Tensordot/Prod_1
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_4/Tensordot/concat/axis
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_4/Tensordot/concatи
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/stackп
&sequential/dense_4/Tensordot/transpose	Transposelayer_normalization/add:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2(
&sequential/dense_4/Tensordot/transposeы
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_4/Tensordot/Reshapeы
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#sequential/dense_4/Tensordot/MatMul
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/dense_4/Tensordot/Const_2
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/concat_1/axis
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/concat_1н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/TensordotЦ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)sequential/dense_4/BiasAdd/ReadVariableOpд
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/BiasAddІ
sequential/dense_4/p_re_lu/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22!
sequential/dense_4/p_re_lu/ReluЪ
)sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOp2sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02+
)sequential/dense_4/p_re_lu/ReadVariableOpЄ
sequential/dense_4/p_re_lu/NegNeg1sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22 
sequential/dense_4/p_re_lu/NegЇ
 sequential/dense_4/p_re_lu/Neg_1Neg#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22"
 sequential/dense_4/p_re_lu/Neg_1Ћ
!sequential/dense_4/p_re_lu/Relu_1Relu$sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22#
!sequential/dense_4/p_re_lu/Relu_1г
sequential/dense_4/p_re_lu/mulMul"sequential/dense_4/p_re_lu/Neg:y:0/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/mulг
sequential/dense_4/p_re_lu/addAddV2-sequential/dense_4/p_re_lu/Relu:activations:0"sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/addб
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_5/Tensordot/ReadVariableOp
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_5/Tensordot/axes
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_5/Tensordot/free
"sequential/dense_5/Tensordot/ShapeShape"sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/Shape
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/GatherV2/axisА
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/GatherV2
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_5/Tensordot/GatherV2_1/axisЖ
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_5/Tensordot/GatherV2_1
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_5/Tensordot/ConstЬ
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_5/Tensordot/Prod
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_5/Tensordot/Const_1д
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_5/Tensordot/Prod_1
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_5/Tensordot/concat/axis
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_5/Tensordot/concatи
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/stackц
&sequential/dense_5/Tensordot/transpose	Transpose"sequential/dense_4/p_re_lu/add:z:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22(
&sequential/dense_5/Tensordot/transposeы
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_5/Tensordot/Reshapeы
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2%
#sequential/dense_5/Tensordot/MatMul
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2&
$sequential/dense_5/Tensordot/Const_2
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/concat_1/axis
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/concat_1н
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/TensordotЦ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02+
)sequential/dense_5/BiasAdd/ReadVariableOpд
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/BiasAdd
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/Identity
add_1AddV2layer_normalization/add:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
add_1s
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape 
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stackЄ
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1Є
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2ц
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/xВ
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mulЄ
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stackЈ
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1Ј
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2№
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1Б
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1Є
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_2/stackЈ
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_1Ј
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_2№
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_2
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_2/xК
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_2
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3Ђ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shapeХ
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization_1/Const
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dimsО
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dimsЦ
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill_1
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3н
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2(
&layer_normalization_1/FusedBatchNormV3ж
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2!
layer_normalization_1/Reshape_1Щ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02,
*layer_normalization_1/mul_3/ReadVariableOpж
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/mul_3У
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization_1/add/ReadVariableOpЩ
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/addv
IdentityIdentitylayer_normalization_1/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
'
Щ
G__inference_functional_1_layer_call_and_return_conditional_losses_65236
input_1
transformer_block_65184
transformer_block_65186
transformer_block_65188
transformer_block_65190
transformer_block_65192
transformer_block_65194
transformer_block_65196
transformer_block_65198
transformer_block_65200
transformer_block_65202
transformer_block_65204
transformer_block_65206
transformer_block_65208
transformer_block_65210
transformer_block_65212
transformer_block_65214
transformer_block_65216
dense_6_65221
dense_6_65223
p_re_lu_1_65226
dense_7_65230
dense_7_65232
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ!p_re_lu_1/StatefulPartitionedCallЂ)transformer_block/StatefulPartitionedCallй
)transformer_block/StatefulPartitionedCallStatefulPartitionedCallinput_1transformer_block_65184transformer_block_65186transformer_block_65188transformer_block_65190transformer_block_65192transformer_block_65194transformer_block_65196transformer_block_65198transformer_block_65200transformer_block_65202transformer_block_65204transformer_block_65206transformer_block_65208transformer_block_65210transformer_block_65212transformer_block_65214transformer_block_65216*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_649372+
)transformer_block/StatefulPartitionedCallА
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_650572*
(global_average_pooling1d/PartitionedCall
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650812
dropout_2/PartitionedCallЉ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_6_65221dense_6_65223*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_651042!
dense_6/StatefulPartitionedCallІ
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0p_re_lu_1_65226*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_643092#
!p_re_lu_1/StatefulPartitionedCallћ
dropout_3/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651402
dropout_3/PartitionedCallЈ
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_65230dense_7_65232*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_651642!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1
м
|
'__inference_dense_6_layer_call_fn_67044

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_651042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџХ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs
%
в
B__inference_dense_4_layer_call_and_return_conditional_losses_64131

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22	
BiasAddm
p_re_lu/ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Relu
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02
p_re_lu/ReadVariableOpk
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
p_re_lu/Negn
p_re_lu/Neg_1NegBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Neg_1r
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/Relu_1
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/mul
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22
p_re_lu/addh
IdentityIdentityp_re_lu/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ2Х::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs

Й
,__inference_functional_1_layer_call_fn_65341
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_652942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1
 
Ѕ
G__inference_functional_1_layer_call_and_return_conditional_losses_65858

inputsW
Stransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resourceU
Qtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceG
Ctransformer_block_layer_normalization_mul_3_readvariableop_resourceE
Atransformer_block_layer_normalization_add_readvariableop_resourceJ
Ftransformer_block_sequential_dense_4_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_biasadd_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_p_re_lu_readvariableop_resourceJ
Ftransformer_block_sequential_dense_5_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_5_biasadd_readvariableop_resourceI
Etransformer_block_layer_normalization_1_mul_3_readvariableop_resourceG
Ctransformer_block_layer_normalization_1_add_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity
1transformer_block/multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:23
1transformer_block/multi_head_self_attention/ShapeЬ
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?transformer_block/multi_head_self_attention/strided_slice/stackа
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Atransformer_block/multi_head_self_attention/strided_slice/stack_1а
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Atransformer_block/multi_head_self_attention/strided_slice/stack_2ъ
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9transformer_block/multi_head_self_attention/strided_sliceЎ
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02L
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpЮ
@transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block/multi_head_self_attention/dense/Tensordot/axesе
@transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@transformer_block/multi_head_self_attention/dense/Tensordot/freeМ
Atransformer_block/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/Shapeи
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisЫ
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2м
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisб
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1а
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstШ
@transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@transformer_block/multi_head_self_attention/dense/Tensordot/Prodд
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1а
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1д
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisЊ
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatд
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackPackItransformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackЇ
Etransformer_block/multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsKtransformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2G
Etransformer_block/multi_head_self_attention/dense/Tensordot/transposeч
Ctransformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Reshapeч
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulе
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2и
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisЗ
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1й
;transformer_block/multi_head_self_attention/dense/TensordotReshapeLtransformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense/TensordotЃ
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02J
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpа
9transformer_block/multi_head_self_attention/dense/BiasAddBiasAddDtransformer_block/multi_head_self_attention/dense/Tensordot:output:0Ptransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2;
9transformer_block/multi_head_self_attention/dense/BiasAddД
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeР
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stack­
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsMtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2I
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1с
=transformer_block/multi_head_self_attention/dense_1/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2?
=transformer_block/multi_head_self_attention/dense_1/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpи
;transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_1/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense_1/BiasAddД
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeР
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stack­
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsMtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2I
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1с
=transformer_block/multi_head_self_attention/dense_2/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2?
=transformer_block/multi_head_self_attention/dense_2/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpи
;transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_2/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense_2/BiasAddХ
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2=
;transformer_block/multi_head_self_attention/Reshape/shape/1М
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2=
;transformer_block/multi_head_self_attention/Reshape/shape/2М
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2=
;transformer_block/multi_head_self_attention/Reshape/shape/3Т
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_self_attention/Reshape/shapeР
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/dense/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/Reshapeб
:transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2<
:transformer_block/multi_head_self_attention/transpose/permС
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/transposeЩ
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Р
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Р
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Ь
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_1/shapeШ
5transformer_block/multi_head_self_attention/Reshape_1ReshapeDtransformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/Reshape_1е
<transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_1/permЩ
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_1Щ
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Р
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Р
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Ь
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_2/shapeШ
5transformer_block/multi_head_self_attention/Reshape_2ReshapeDtransformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/Reshape_2е
<transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_2/permЩ
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_2Ъ
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(24
2transformer_block/multi_head_self_attention/MatMulе
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:25
3transformer_block/multi_head_self_attention/Shape_1й
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2C
Atransformer_block/multi_head_self_attention/strided_slice_1/stackд
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1д
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2і
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;transformer_block/multi_head_self_attention/strided_slice_1т
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0transformer_block/multi_head_self_attention/CastУ
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 22
0transformer_block/multi_head_self_attention/SqrtД
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/truedivњ
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/SoftmaxМ
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ26
4transformer_block/multi_head_self_attention/MatMul_1е
<transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_3/permШ
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_3Щ
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_3/shape/1С
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2?
=transformer_block/multi_head_self_attention/Reshape_3/shape/2
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_3/shapeМ
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5transformer_block/multi_head_self_attention/Reshape_3Д
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeј
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackю
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2I
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ъ
=transformer_block/multi_head_self_attention/dense_3/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2?
=transformer_block/multi_head_self_attention/dense_3/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpс
;transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_3/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2=
;transformer_block/multi_head_self_attention/dense_3/BiasAdd
'transformer_block/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2)
'transformer_block/dropout/dropout/Const
%transformer_block/dropout/dropout/MulMulDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:00transformer_block/dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2'
%transformer_block/dropout/dropout/MulЦ
'transformer_block/dropout/dropout/ShapeShapeDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2)
'transformer_block/dropout/dropout/Shape
>transformer_block/dropout/dropout/random_uniform/RandomUniformRandomUniform0transformer_block/dropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ*
dtype02@
>transformer_block/dropout/dropout/random_uniform/RandomUniformЉ
0transformer_block/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0transformer_block/dropout/dropout/GreaterEqual/yД
.transformer_block/dropout/dropout/GreaterEqualGreaterEqualGtransformer_block/dropout/dropout/random_uniform/RandomUniform:output:09transformer_block/dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ20
.transformer_block/dropout/dropout/GreaterEqualл
&transformer_block/dropout/dropout/CastCast2transformer_block/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2(
&transformer_block/dropout/dropout/Cast№
'transformer_block/dropout/dropout/Mul_1Mul)transformer_block/dropout/dropout/Mul:z:0*transformer_block/dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2)
'transformer_block/dropout/dropout/Mul_1Ѓ
transformer_block/addAddV2inputs+transformer_block/dropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
transformer_block/addЃ
+transformer_block/layer_normalization/ShapeShapetransformer_block/add:z:0*
T0*
_output_shapes
:2-
+transformer_block/layer_normalization/ShapeР
9transformer_block/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block/layer_normalization/strided_slice/stackФ
;transformer_block/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice/stack_1Ф
;transformer_block/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice/stack_2Ц
3transformer_block/layer_normalization/strided_sliceStridedSlice4transformer_block/layer_normalization/Shape:output:0Btransformer_block/layer_normalization/strided_slice/stack:output:0Dtransformer_block/layer_normalization/strided_slice/stack_1:output:0Dtransformer_block/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3transformer_block/layer_normalization/strided_slice
+transformer_block/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2-
+transformer_block/layer_normalization/mul/xђ
)transformer_block/layer_normalization/mulMul4transformer_block/layer_normalization/mul/x:output:0<transformer_block/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2+
)transformer_block/layer_normalization/mulФ
;transformer_block/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice_1/stackШ
=transformer_block/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_1/stack_1Ш
=transformer_block/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_1/stack_2а
5transformer_block/layer_normalization/strided_slice_1StridedSlice4transformer_block/layer_normalization/Shape:output:0Dtransformer_block/layer_normalization/strided_slice_1/stack:output:0Ftransformer_block/layer_normalization/strided_slice_1/stack_1:output:0Ftransformer_block/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization/strided_slice_1ё
+transformer_block/layer_normalization/mul_1Mul-transformer_block/layer_normalization/mul:z:0>transformer_block/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization/mul_1Ф
;transformer_block/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice_2/stackШ
=transformer_block/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_2/stack_1Ш
=transformer_block/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_2/stack_2а
5transformer_block/layer_normalization/strided_slice_2StridedSlice4transformer_block/layer_normalization/Shape:output:0Dtransformer_block/layer_normalization/strided_slice_2/stack:output:0Ftransformer_block/layer_normalization/strided_slice_2/stack_1:output:0Ftransformer_block/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization/strided_slice_2 
-transformer_block/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2/
-transformer_block/layer_normalization/mul_2/xњ
+transformer_block/layer_normalization/mul_2Mul6transformer_block/layer_normalization/mul_2/x:output:0>transformer_block/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization/mul_2А
5transformer_block/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :27
5transformer_block/layer_normalization/Reshape/shape/0А
5transformer_block/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :27
5transformer_block/layer_normalization/Reshape/shape/3
3transformer_block/layer_normalization/Reshape/shapePack>transformer_block/layer_normalization/Reshape/shape/0:output:0/transformer_block/layer_normalization/mul_1:z:0/transformer_block/layer_normalization/mul_2:z:0>transformer_block/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3transformer_block/layer_normalization/Reshape/shape
-transformer_block/layer_normalization/ReshapeReshapetransformer_block/add:z:0<transformer_block/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2/
-transformer_block/layer_normalization/Reshape
+transformer_block/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+transformer_block/layer_normalization/ConstЩ
/transformer_block/layer_normalization/Fill/dimsPack/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:21
/transformer_block/layer_normalization/Fill/dimsў
*transformer_block/layer_normalization/FillFill8transformer_block/layer_normalization/Fill/dims:output:04transformer_block/layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2,
*transformer_block/layer_normalization/FillЃ
-transformer_block/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2/
-transformer_block/layer_normalization/Const_1Э
1transformer_block/layer_normalization/Fill_1/dimsPack/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:23
1transformer_block/layer_normalization/Fill_1/dims
,transformer_block/layer_normalization/Fill_1Fill:transformer_block/layer_normalization/Fill_1/dims:output:06transformer_block/layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,transformer_block/layer_normalization/Fill_1Ё
-transformer_block/layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2/
-transformer_block/layer_normalization/Const_2Ё
-transformer_block/layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2/
-transformer_block/layer_normalization/Const_3Э
6transformer_block/layer_normalization/FusedBatchNormV3FusedBatchNormV36transformer_block/layer_normalization/Reshape:output:03transformer_block/layer_normalization/Fill:output:05transformer_block/layer_normalization/Fill_1:output:06transformer_block/layer_normalization/Const_2:output:06transformer_block/layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:28
6transformer_block/layer_normalization/FusedBatchNormV3
/transformer_block/layer_normalization/Reshape_1Reshape:transformer_block/layer_normalization/FusedBatchNormV3:y:04transformer_block/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х21
/transformer_block/layer_normalization/Reshape_1љ
:transformer_block/layer_normalization/mul_3/ReadVariableOpReadVariableOpCtransformer_block_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02<
:transformer_block/layer_normalization/mul_3/ReadVariableOp
+transformer_block/layer_normalization/mul_3Mul8transformer_block/layer_normalization/Reshape_1:output:0Btransformer_block/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+transformer_block/layer_normalization/mul_3ѓ
8transformer_block/layer_normalization/add/ReadVariableOpReadVariableOpAtransformer_block_layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8transformer_block/layer_normalization/add/ReadVariableOp
)transformer_block/layer_normalization/addAddV2/transformer_block/layer_normalization/mul_3:z:0@transformer_block/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)transformer_block/layer_normalization/add
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02?
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpД
3transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3transformer_block/sequential/dense_4/Tensordot/axesЛ
3transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3transformer_block/sequential/dense_4/Tensordot/freeЩ
4transformer_block/sequential/dense_4/Tensordot/ShapeShape-transformer_block/layer_normalization/add:z:0*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_4/Tensordot/ShapeО
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axis
7transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/free:output:0Etransformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7transformer_block/sequential/dense_4/Tensordot/GatherV2Т
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Gtransformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1Ж
4transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/sequential/dense_4/Tensordot/Const
3transformer_block/sequential/dense_4/Tensordot/ProdProd@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3transformer_block/sequential/dense_4/Tensordot/ProdК
6transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/sequential/dense_4/Tensordot/Const_1
5transformer_block/sequential/dense_4/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5transformer_block/sequential/dense_4/Tensordot/Prod_1К
:transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block/sequential/dense_4/Tensordot/concat/axisщ
5transformer_block/sequential/dense_4/Tensordot/concatConcatV2<transformer_block/sequential/dense_4/Tensordot/free:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Ctransformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_4/Tensordot/concat 
4transformer_block/sequential/dense_4/Tensordot/stackPack<transformer_block/sequential/dense_4/Tensordot/Prod:output:0>transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_4/Tensordot/stackЇ
8transformer_block/sequential/dense_4/Tensordot/transpose	Transpose-transformer_block/layer_normalization/add:z:0>transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2:
8transformer_block/sequential/dense_4/Tensordot/transposeГ
6transformer_block/sequential/dense_4/Tensordot/ReshapeReshape<transformer_block/sequential/dense_4/Tensordot/transpose:y:0=transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6transformer_block/sequential/dense_4/Tensordot/ReshapeГ
5transformer_block/sequential/dense_4/Tensordot/MatMulMatMul?transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ27
5transformer_block/sequential/dense_4/Tensordot/MatMulЛ
6transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/sequential/dense_4/Tensordot/Const_2О
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisі
7transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/sequential/dense_4/Tensordot/concat_1Ѕ
.transformer_block/sequential/dense_4/TensordotReshape?transformer_block/sequential/dense_4/Tensordot/MatMul:product:0@transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ220
.transformer_block/sequential/dense_4/Tensordotќ
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp
,transformer_block/sequential/dense_4/BiasAddBiasAdd7transformer_block/sequential/dense_4/Tensordot:output:0Ctransformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22.
,transformer_block/sequential/dense_4/BiasAddм
1transformer_block/sequential/dense_4/p_re_lu/ReluRelu5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ223
1transformer_block/sequential/dense_4/p_re_lu/Relu
;transformer_block/sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02=
;transformer_block/sequential/dense_4/p_re_lu/ReadVariableOpк
0transformer_block/sequential/dense_4/p_re_lu/NegNegCtransformer_block/sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	222
0transformer_block/sequential/dense_4/p_re_lu/Negн
2transformer_block/sequential/dense_4/p_re_lu/Neg_1Neg5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ224
2transformer_block/sequential/dense_4/p_re_lu/Neg_1с
3transformer_block/sequential/dense_4/p_re_lu/Relu_1Relu6transformer_block/sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ225
3transformer_block/sequential/dense_4/p_re_lu/Relu_1
0transformer_block/sequential/dense_4/p_re_lu/mulMul4transformer_block/sequential/dense_4/p_re_lu/Neg:y:0Atransformer_block/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ222
0transformer_block/sequential/dense_4/p_re_lu/mul
0transformer_block/sequential/dense_4/p_re_lu/addAddV2?transformer_block/sequential/dense_4/p_re_lu/Relu:activations:04transformer_block/sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ222
0transformer_block/sequential/dense_4/p_re_lu/add
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02?
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpД
3transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3transformer_block/sequential/dense_5/Tensordot/axesЛ
3transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3transformer_block/sequential/dense_5/Tensordot/freeа
4transformer_block/sequential/dense_5/Tensordot/ShapeShape4transformer_block/sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_5/Tensordot/ShapeО
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axis
7transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/free:output:0Etransformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7transformer_block/sequential/dense_5/Tensordot/GatherV2Т
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Gtransformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1Ж
4transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/sequential/dense_5/Tensordot/Const
3transformer_block/sequential/dense_5/Tensordot/ProdProd@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3transformer_block/sequential/dense_5/Tensordot/ProdК
6transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/sequential/dense_5/Tensordot/Const_1
5transformer_block/sequential/dense_5/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5transformer_block/sequential/dense_5/Tensordot/Prod_1К
:transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block/sequential/dense_5/Tensordot/concat/axisщ
5transformer_block/sequential/dense_5/Tensordot/concatConcatV2<transformer_block/sequential/dense_5/Tensordot/free:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Ctransformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_5/Tensordot/concat 
4transformer_block/sequential/dense_5/Tensordot/stackPack<transformer_block/sequential/dense_5/Tensordot/Prod:output:0>transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_5/Tensordot/stackЎ
8transformer_block/sequential/dense_5/Tensordot/transpose	Transpose4transformer_block/sequential/dense_4/p_re_lu/add:z:0>transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22:
8transformer_block/sequential/dense_5/Tensordot/transposeГ
6transformer_block/sequential/dense_5/Tensordot/ReshapeReshape<transformer_block/sequential/dense_5/Tensordot/transpose:y:0=transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6transformer_block/sequential/dense_5/Tensordot/ReshapeГ
5transformer_block/sequential/dense_5/Tensordot/MatMulMatMul?transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ27
5transformer_block/sequential/dense_5/Tensordot/MatMulЛ
6transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х28
6transformer_block/sequential/dense_5/Tensordot/Const_2О
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisі
7transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/sequential/dense_5/Tensordot/concat_1Ѕ
.transformer_block/sequential/dense_5/TensordotReshape?transformer_block/sequential/dense_5/Tensordot/MatMul:product:0@transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х20
.transformer_block/sequential/dense_5/Tensordotќ
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02=
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp
,transformer_block/sequential/dense_5/BiasAddBiasAdd7transformer_block/sequential/dense_5/Tensordot:output:0Ctransformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2.
,transformer_block/sequential/dense_5/BiasAdd
)transformer_block/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)transformer_block/dropout_1/dropout/Constћ
'transformer_block/dropout_1/dropout/MulMul5transformer_block/sequential/dense_5/BiasAdd:output:02transformer_block/dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2)
'transformer_block/dropout_1/dropout/MulЛ
)transformer_block/dropout_1/dropout/ShapeShape5transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2+
)transformer_block/dropout_1/dropout/Shape
@transformer_block/dropout_1/dropout/random_uniform/RandomUniformRandomUniform2transformer_block/dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х*
dtype02B
@transformer_block/dropout_1/dropout/random_uniform/RandomUniform­
2transformer_block/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?24
2transformer_block/dropout_1/dropout/GreaterEqual/yГ
0transformer_block/dropout_1/dropout/GreaterEqualGreaterEqualItransformer_block/dropout_1/dropout/random_uniform/RandomUniform:output:0;transformer_block/dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х22
0transformer_block/dropout_1/dropout/GreaterEqualи
(transformer_block/dropout_1/dropout/CastCast4transformer_block/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ2Х2*
(transformer_block/dropout_1/dropout/Castя
)transformer_block/dropout_1/dropout/Mul_1Mul+transformer_block/dropout_1/dropout/Mul:z:0,transformer_block/dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)transformer_block/dropout_1/dropout/Mul_1а
transformer_block/add_1AddV2-transformer_block/layer_normalization/add:z:0-transformer_block/dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
transformer_block/add_1Љ
-transformer_block/layer_normalization_1/ShapeShapetransformer_block/add_1:z:0*
T0*
_output_shapes
:2/
-transformer_block/layer_normalization_1/ShapeФ
;transformer_block/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block/layer_normalization_1/strided_slice/stackШ
=transformer_block/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice/stack_1Ш
=transformer_block/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice/stack_2в
5transformer_block/layer_normalization_1/strided_sliceStridedSlice6transformer_block/layer_normalization_1/Shape:output:0Dtransformer_block/layer_normalization_1/strided_slice/stack:output:0Ftransformer_block/layer_normalization_1/strided_slice/stack_1:output:0Ftransformer_block/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization_1/strided_slice 
-transformer_block/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2/
-transformer_block/layer_normalization_1/mul/xњ
+transformer_block/layer_normalization_1/mulMul6transformer_block/layer_normalization_1/mul/x:output:0>transformer_block/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization_1/mulШ
=transformer_block/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice_1/stackЬ
?transformer_block/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_1/stack_1Ь
?transformer_block/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_1/stack_2м
7transformer_block/layer_normalization_1/strided_slice_1StridedSlice6transformer_block/layer_normalization_1/Shape:output:0Ftransformer_block/layer_normalization_1/strided_slice_1/stack:output:0Htransformer_block/layer_normalization_1/strided_slice_1/stack_1:output:0Htransformer_block/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7transformer_block/layer_normalization_1/strided_slice_1љ
-transformer_block/layer_normalization_1/mul_1Mul/transformer_block/layer_normalization_1/mul:z:0@transformer_block/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2/
-transformer_block/layer_normalization_1/mul_1Ш
=transformer_block/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice_2/stackЬ
?transformer_block/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_2/stack_1Ь
?transformer_block/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_2/stack_2м
7transformer_block/layer_normalization_1/strided_slice_2StridedSlice6transformer_block/layer_normalization_1/Shape:output:0Ftransformer_block/layer_normalization_1/strided_slice_2/stack:output:0Htransformer_block/layer_normalization_1/strided_slice_2/stack_1:output:0Htransformer_block/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7transformer_block/layer_normalization_1/strided_slice_2Є
/transformer_block/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :21
/transformer_block/layer_normalization_1/mul_2/x
-transformer_block/layer_normalization_1/mul_2Mul8transformer_block/layer_normalization_1/mul_2/x:output:0@transformer_block/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2/
-transformer_block/layer_normalization_1/mul_2Д
7transformer_block/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :29
7transformer_block/layer_normalization_1/Reshape/shape/0Д
7transformer_block/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :29
7transformer_block/layer_normalization_1/Reshape/shape/3
5transformer_block/layer_normalization_1/Reshape/shapePack@transformer_block/layer_normalization_1/Reshape/shape/0:output:01transformer_block/layer_normalization_1/mul_1:z:01transformer_block/layer_normalization_1/mul_2:z:0@transformer_block/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/layer_normalization_1/Reshape/shape
/transformer_block/layer_normalization_1/ReshapeReshapetransformer_block/add_1:z:0>transformer_block/layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ21
/transformer_block/layer_normalization_1/ReshapeЃ
-transformer_block/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-transformer_block/layer_normalization_1/ConstЯ
1transformer_block/layer_normalization_1/Fill/dimsPack1transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:23
1transformer_block/layer_normalization_1/Fill/dims
,transformer_block/layer_normalization_1/FillFill:transformer_block/layer_normalization_1/Fill/dims:output:06transformer_block/layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,transformer_block/layer_normalization_1/FillЇ
/transformer_block/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    21
/transformer_block/layer_normalization_1/Const_1г
3transformer_block/layer_normalization_1/Fill_1/dimsPack1transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:25
3transformer_block/layer_normalization_1/Fill_1/dims
.transformer_block/layer_normalization_1/Fill_1Fill<transformer_block/layer_normalization_1/Fill_1/dims:output:08transformer_block/layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ20
.transformer_block/layer_normalization_1/Fill_1Ѕ
/transformer_block/layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 21
/transformer_block/layer_normalization_1/Const_2Ѕ
/transformer_block/layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 21
/transformer_block/layer_normalization_1/Const_3л
8transformer_block/layer_normalization_1/FusedBatchNormV3FusedBatchNormV38transformer_block/layer_normalization_1/Reshape:output:05transformer_block/layer_normalization_1/Fill:output:07transformer_block/layer_normalization_1/Fill_1:output:08transformer_block/layer_normalization_1/Const_2:output:08transformer_block/layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2:
8transformer_block/layer_normalization_1/FusedBatchNormV3
1transformer_block/layer_normalization_1/Reshape_1Reshape<transformer_block/layer_normalization_1/FusedBatchNormV3:y:06transformer_block/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х23
1transformer_block/layer_normalization_1/Reshape_1џ
<transformer_block/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpEtransformer_block_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02>
<transformer_block/layer_normalization_1/mul_3/ReadVariableOp
-transformer_block/layer_normalization_1/mul_3Mul:transformer_block/layer_normalization_1/Reshape_1:output:0Dtransformer_block/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2/
-transformer_block/layer_normalization_1/mul_3љ
:transformer_block/layer_normalization_1/add/ReadVariableOpReadVariableOpCtransformer_block_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02<
:transformer_block/layer_normalization_1/add/ReadVariableOp
+transformer_block/layer_normalization_1/addAddV21transformer_block/layer_normalization_1/mul_3:z:0Btransformer_block/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+transformer_block/layer_normalization_1/addЄ
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesф
global_average_pooling1d/MeanMean/transformer_block/layer_normalization_1/add:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
global_average_pooling1d/Meanw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_2/dropout/ConstВ
dropout_2/dropout/MulMul&global_average_pooling1d/Mean:output:0 dropout_2/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout_2/dropout/Mul
dropout_2/dropout/ShapeShape&global_average_pooling1d/Mean:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/Shapeг
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_2/dropout/GreaterEqual/yч
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2 
dropout_2/dropout/GreaterEqual
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџХ2
dropout_2/dropout/CastЃ
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout_2/dropout/Mul_1Ї
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
dense_6/MatMul/ReadVariableOpЁ
dense_6/MatMulMatMuldropout_2/dropout/Mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_6/MatMulЅ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpЂ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_6/BiasAddu
p_re_lu_1/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Relu
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*
_output_shapes	
:*
dtype02
p_re_lu_1/ReadVariableOpm
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
p_re_lu_1/Negv
p_re_lu_1/Neg_1Negdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Neg_1t
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Relu_1
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/mul
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/addw
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_3/dropout/Const
dropout_3/dropout/MulMulp_re_lu_1/add:z:0 dropout_3/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Muls
dropout_3/dropout/ShapeShapep_re_lu_1/add:z:0*
T0*
_output_shapes
:2
dropout_3/dropout/Shapeг
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_3/dropout/GreaterEqual/yч
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2 
dropout_3/dropout/GreaterEqual
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/CastЃ
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/dropout/Mul_1І
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldropout_3/dropout/Mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/MatMulЄ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_7/BiasAdd/ReadVariableOpЁ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/Softmaxm
IdentityIdentitydense_7/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х:::::::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs


'__inference_dense_4_layer_call_fn_67298

inputs
unknown
	unknown_0
	unknown_1
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_641312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџ2Х:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
*

G__inference_functional_1_layer_call_and_return_conditional_losses_65181
input_1
transformer_block_65017
transformer_block_65019
transformer_block_65021
transformer_block_65023
transformer_block_65025
transformer_block_65027
transformer_block_65029
transformer_block_65031
transformer_block_65033
transformer_block_65035
transformer_block_65037
transformer_block_65039
transformer_block_65041
transformer_block_65043
transformer_block_65045
transformer_block_65047
transformer_block_65049
dense_6_65115
dense_6_65117
p_re_lu_1_65120
dense_7_65175
dense_7_65177
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallЂ!p_re_lu_1/StatefulPartitionedCallЂ)transformer_block/StatefulPartitionedCallй
)transformer_block/StatefulPartitionedCallStatefulPartitionedCallinput_1transformer_block_65017transformer_block_65019transformer_block_65021transformer_block_65023transformer_block_65025transformer_block_65027transformer_block_65029transformer_block_65031transformer_block_65033transformer_block_65035transformer_block_65037transformer_block_65039transformer_block_65041transformer_block_65043transformer_block_65045transformer_block_65047transformer_block_65049*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_646362+
)transformer_block/StatefulPartitionedCallА
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_650572*
(global_average_pooling1d/PartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650762#
!dropout_2/StatefulPartitionedCallБ
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_6_65115dense_6_65117*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_651042!
dense_6/StatefulPartitionedCallІ
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0p_re_lu_1_65120*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_643092#
!p_re_lu_1/StatefulPartitionedCallЗ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651352#
!dropout_3/StatefulPartitionedCallА
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_65175dense_7_65177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_651642!
dense_7/StatefulPartitionedCallи
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1
ЁL
Щ
E__inference_sequential_layer_call_and_return_conditional_losses_67219

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource+
'dense_4_p_re_lu_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityА
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/freeh
dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisљ
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axisџ
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1Ј
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisи
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatЌ
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stackЉ
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_4/Tensordot/transposeП
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_4/Tensordot/ReshapeП
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/Tensordot/MatMul
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisх
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1Б
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/TensordotЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЈ
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/BiasAdd
dense_4/p_re_lu/ReluReludense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/ReluЉ
dense_4/p_re_lu/ReadVariableOpReadVariableOp'dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02 
dense_4/p_re_lu/ReadVariableOp
dense_4/p_re_lu/NegNeg&dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
dense_4/p_re_lu/Neg
dense_4/p_re_lu/Neg_1Negdense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/Neg_1
dense_4/p_re_lu/Relu_1Reludense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/Relu_1Ї
dense_4/p_re_lu/mulMuldense_4/p_re_lu/Neg:y:0$dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/mulЇ
dense_4/p_re_lu/addAddV2"dense_4/p_re_lu/Relu:activations:0dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/addА
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/freey
dense_5/Tensordot/ShapeShapedense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisљ
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axisџ
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1Ј
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisи
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concatЌ
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stackК
dense_5/Tensordot/transpose	Transposedense_4/p_re_lu/add:z:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_5/Tensordot/transposeП
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_5/Tensordot/ReshapeП
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisх
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1Б
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_5/TensordotЅ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02 
dense_5/BiasAdd/ReadVariableOpЈ
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_5/BiasAddq
IdentityIdentitydense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
ш
Г
*__inference_sequential_layer_call_fn_64277
dense_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCalldense_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_642642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::22
StatefulPartitionedCallStatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ2Х
'
_user_specified_namedense_4_input
ю
T
8__inference_global_average_pooling1d_layer_call_fn_66998

inputs
identityк
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_642932
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
э

э
1__inference_transformer_block_layer_call_fn_66937

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_646362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_65135

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д

D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_64309

inputs
readvariableop_resource
identityW
ReluReluinputs*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Reluu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOpO
NegNegReadVariableOp:value:0*
T0*
_output_shapes	
:2
NegX
Neg_1Neginputs*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Neg_1^
Relu_1Relu	Neg_1:y:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Relu_1c
mulMulNeg:y:0Relu_1:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
mulc
addAddV2Relu:activations:0mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
add\
IdentityIdentityadd:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ::X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

И
,__inference_functional_1_layer_call_fn_66233

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_652942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
я
Ѕ
G__inference_functional_1_layer_call_and_return_conditional_losses_66184

inputsW
Stransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resourceU
Qtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resourceY
Utransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resourceW
Stransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceG
Ctransformer_block_layer_normalization_mul_3_readvariableop_resourceE
Atransformer_block_layer_normalization_add_readvariableop_resourceJ
Ftransformer_block_sequential_dense_4_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_biasadd_readvariableop_resourceH
Dtransformer_block_sequential_dense_4_p_re_lu_readvariableop_resourceJ
Ftransformer_block_sequential_dense_5_tensordot_readvariableop_resourceH
Dtransformer_block_sequential_dense_5_biasadd_readvariableop_resourceI
Etransformer_block_layer_normalization_1_mul_3_readvariableop_resourceG
Ctransformer_block_layer_normalization_1_add_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identity
1transformer_block/multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:23
1transformer_block/multi_head_self_attention/ShapeЬ
?transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?transformer_block/multi_head_self_attention/strided_slice/stackа
Atransformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Atransformer_block/multi_head_self_attention/strided_slice/stack_1а
Atransformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Atransformer_block/multi_head_self_attention/strided_slice/stack_2ъ
9transformer_block/multi_head_self_attention/strided_sliceStridedSlice:transformer_block/multi_head_self_attention/Shape:output:0Htransformer_block/multi_head_self_attention/strided_slice/stack:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9transformer_block/multi_head_self_attention/strided_sliceЎ
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02L
Jtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpЮ
@transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@transformer_block/multi_head_self_attention/dense/Tensordot/axesе
@transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@transformer_block/multi_head_self_attention/dense/Tensordot/freeМ
Atransformer_block/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/Shapeи
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisЫ
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2м
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisб
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Jtransformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ttransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1а
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/ConstШ
@transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdMtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@transformer_block/multi_head_self_attention/dense/Tensordot/Prodд
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_1а
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ProdOtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1д
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gtransformer_block/multi_head_self_attention/dense/Tensordot/concat/axisЊ
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Itransformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Itransformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0Ptransformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/concatд
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackPackItransformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Ktransformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Atransformer_block/multi_head_self_attention/dense/Tensordot/stackЇ
Etransformer_block/multi_head_self_attention/dense/Tensordot/transpose	TransposeinputsKtransformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2G
Etransformer_block/multi_head_self_attention/dense/Tensordot/transposeч
Ctransformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeItransformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Jtransformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Reshapeч
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulLtransformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2D
Btransformer_block/multi_head_self_attention/dense/Tensordot/MatMulе
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2E
Ctransformer_block/multi_head_self_attention/dense/Tensordot/Const_2и
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisЗ
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Mtransformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0Rtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1й
;transformer_block/multi_head_self_attention/dense/TensordotReshapeLtransformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Mtransformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense/TensordotЃ
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOpQtransformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02J
Htransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpа
9transformer_block/multi_head_self_attention/dense/BiasAddBiasAddDtransformer_block/multi_head_self_attention/dense/Tensordot:output:0Ptransformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2;
9transformer_block/multi_head_self_attention/dense/BiasAddД
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/freeР
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_1/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_1/Tensordot/stack­
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	TransposeinputsMtransformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2I
Gtransformer_block/multi_head_self_attention/dense_1/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1с
=transformer_block/multi_head_self_attention/dense_1/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2?
=transformer_block/multi_head_self_attention/dense_1/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpи
;transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_1/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense_1/BiasAddД
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/freeР
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_2/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_2/Tensordot/stack­
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	TransposeinputsMtransformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2I
Gtransformer_block/multi_head_self_attention/dense_2/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1с
=transformer_block/multi_head_self_attention/dense_2/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2?
=transformer_block/multi_head_self_attention/dense_2/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpи
;transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_2/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;transformer_block/multi_head_self_attention/dense_2/BiasAddХ
;transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2=
;transformer_block/multi_head_self_attention/Reshape/shape/1М
;transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2=
;transformer_block/multi_head_self_attention/Reshape/shape/2М
;transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2=
;transformer_block/multi_head_self_attention/Reshape/shape/3Т
9transformer_block/multi_head_self_attention/Reshape/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/1:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/2:output:0Dtransformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2;
9transformer_block/multi_head_self_attention/Reshape/shapeР
3transformer_block/multi_head_self_attention/ReshapeReshapeBtransformer_block/multi_head_self_attention/dense/BiasAdd:output:0Btransformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/Reshapeб
:transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2<
:transformer_block/multi_head_self_attention/transpose/permС
5transformer_block/multi_head_self_attention/transpose	Transpose<transformer_block/multi_head_self_attention/Reshape:output:0Ctransformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/transposeЩ
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/1Р
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/2Р
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_1/shape/3Ь
;transformer_block/multi_head_self_attention/Reshape_1/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_1/shapeШ
5transformer_block/multi_head_self_attention/Reshape_1ReshapeDtransformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/Reshape_1е
<transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_1/permЩ
7transformer_block/multi_head_self_attention/transpose_1	Transpose>transformer_block/multi_head_self_attention/Reshape_1:output:0Etransformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_1Щ
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/1Р
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/2Р
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2?
=transformer_block/multi_head_self_attention/Reshape_2/shape/3Ь
;transformer_block/multi_head_self_attention/Reshape_2/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Ftransformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_2/shapeШ
5transformer_block/multi_head_self_attention/Reshape_2ReshapeDtransformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Dtransformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ27
5transformer_block/multi_head_self_attention/Reshape_2е
<transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_2/permЩ
7transformer_block/multi_head_self_attention/transpose_2	Transpose>transformer_block/multi_head_self_attention/Reshape_2:output:0Etransformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_2Ъ
2transformer_block/multi_head_self_attention/MatMulBatchMatMulV29transformer_block/multi_head_self_attention/transpose:y:0;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(24
2transformer_block/multi_head_self_attention/MatMulе
3transformer_block/multi_head_self_attention/Shape_1Shape;transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:25
3transformer_block/multi_head_self_attention/Shape_1й
Atransformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2C
Atransformer_block/multi_head_self_attention/strided_slice_1/stackд
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_1д
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Ctransformer_block/multi_head_self_attention/strided_slice_1/stack_2і
;transformer_block/multi_head_self_attention/strided_slice_1StridedSlice<transformer_block/multi_head_self_attention/Shape_1:output:0Jtransformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Ltransformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;transformer_block/multi_head_self_attention/strided_slice_1т
0transformer_block/multi_head_self_attention/CastCastDtransformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0transformer_block/multi_head_self_attention/CastУ
0transformer_block/multi_head_self_attention/SqrtSqrt4transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 22
0transformer_block/multi_head_self_attention/SqrtД
3transformer_block/multi_head_self_attention/truedivRealDiv;transformer_block/multi_head_self_attention/MatMul:output:04transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/truedivњ
3transformer_block/multi_head_self_attention/SoftmaxSoftmax7transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ25
3transformer_block/multi_head_self_attention/SoftmaxМ
4transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2=transformer_block/multi_head_self_attention/Softmax:softmax:0;transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ26
4transformer_block/multi_head_self_attention/MatMul_1е
<transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2>
<transformer_block/multi_head_self_attention/transpose_3/permШ
7transformer_block/multi_head_self_attention/transpose_3	Transpose=transformer_block/multi_head_self_attention/MatMul_1:output:0Etransformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ29
7transformer_block/multi_head_self_attention/transpose_3Щ
=transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2?
=transformer_block/multi_head_self_attention/Reshape_3/shape/1С
=transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2?
=transformer_block/multi_head_self_attention/Reshape_3/shape/2
;transformer_block/multi_head_self_attention/Reshape_3/shapePackBtransformer_block/multi_head_self_attention/strided_slice:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Ftransformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2=
;transformer_block/multi_head_self_attention/Reshape_3/shapeМ
5transformer_block/multi_head_self_attention/Reshape_3Reshape;transformer_block/multi_head_self_attention/transpose_3:y:0Dtransformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5transformer_block/multi_head_self_attention/Reshape_3Д
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpUtransformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02N
Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpв
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/axesй
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/freeј
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShape>transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/Shapeм
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisе
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2р
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisл
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Vtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2J
Htransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1д
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/Constа
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProdOtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2D
Btransformer_block/multi_head_self_attention/dense_3/Tensordot/Prodи
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1и
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ProdQtransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1и
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Itransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisД
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0Rtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/concatм
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackKtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2E
Ctransformer_block/multi_head_self_attention/dense_3/Tensordot/stackю
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	Transpose>transformer_block/multi_head_self_attention/Reshape_3:output:0Mtransformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2I
Gtransformer_block/multi_head_self_attention/dense_3/Tensordot/transposeя
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeKtransformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Ltransformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshapeя
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMulNtransformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2F
Dtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulй
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2G
Etransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2м
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Ktransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisС
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2Otransformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Ntransformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Ttransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2H
Ftransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ъ
=transformer_block/multi_head_self_attention/dense_3/TensordotReshapeNtransformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0Otransformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2?
=transformer_block/multi_head_self_attention/dense_3/TensordotЉ
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpStransformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02L
Jtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpс
;transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddFtransformer_block/multi_head_self_attention/dense_3/Tensordot:output:0Rtransformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2=
;transformer_block/multi_head_self_attention/dense_3/BiasAddк
"transformer_block/dropout/IdentityIdentityDtransformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2$
"transformer_block/dropout/IdentityЃ
transformer_block/addAddV2inputs+transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
transformer_block/addЃ
+transformer_block/layer_normalization/ShapeShapetransformer_block/add:z:0*
T0*
_output_shapes
:2-
+transformer_block/layer_normalization/ShapeР
9transformer_block/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9transformer_block/layer_normalization/strided_slice/stackФ
;transformer_block/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice/stack_1Ф
;transformer_block/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice/stack_2Ц
3transformer_block/layer_normalization/strided_sliceStridedSlice4transformer_block/layer_normalization/Shape:output:0Btransformer_block/layer_normalization/strided_slice/stack:output:0Dtransformer_block/layer_normalization/strided_slice/stack_1:output:0Dtransformer_block/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3transformer_block/layer_normalization/strided_slice
+transformer_block/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2-
+transformer_block/layer_normalization/mul/xђ
)transformer_block/layer_normalization/mulMul4transformer_block/layer_normalization/mul/x:output:0<transformer_block/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2+
)transformer_block/layer_normalization/mulФ
;transformer_block/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice_1/stackШ
=transformer_block/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_1/stack_1Ш
=transformer_block/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_1/stack_2а
5transformer_block/layer_normalization/strided_slice_1StridedSlice4transformer_block/layer_normalization/Shape:output:0Dtransformer_block/layer_normalization/strided_slice_1/stack:output:0Ftransformer_block/layer_normalization/strided_slice_1/stack_1:output:0Ftransformer_block/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization/strided_slice_1ё
+transformer_block/layer_normalization/mul_1Mul-transformer_block/layer_normalization/mul:z:0>transformer_block/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization/mul_1Ф
;transformer_block/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;transformer_block/layer_normalization/strided_slice_2/stackШ
=transformer_block/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_2/stack_1Ш
=transformer_block/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization/strided_slice_2/stack_2а
5transformer_block/layer_normalization/strided_slice_2StridedSlice4transformer_block/layer_normalization/Shape:output:0Dtransformer_block/layer_normalization/strided_slice_2/stack:output:0Ftransformer_block/layer_normalization/strided_slice_2/stack_1:output:0Ftransformer_block/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization/strided_slice_2 
-transformer_block/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2/
-transformer_block/layer_normalization/mul_2/xњ
+transformer_block/layer_normalization/mul_2Mul6transformer_block/layer_normalization/mul_2/x:output:0>transformer_block/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization/mul_2А
5transformer_block/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :27
5transformer_block/layer_normalization/Reshape/shape/0А
5transformer_block/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :27
5transformer_block/layer_normalization/Reshape/shape/3
3transformer_block/layer_normalization/Reshape/shapePack>transformer_block/layer_normalization/Reshape/shape/0:output:0/transformer_block/layer_normalization/mul_1:z:0/transformer_block/layer_normalization/mul_2:z:0>transformer_block/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:25
3transformer_block/layer_normalization/Reshape/shape
-transformer_block/layer_normalization/ReshapeReshapetransformer_block/add:z:0<transformer_block/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2/
-transformer_block/layer_normalization/Reshape
+transformer_block/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2-
+transformer_block/layer_normalization/ConstЩ
/transformer_block/layer_normalization/Fill/dimsPack/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:21
/transformer_block/layer_normalization/Fill/dimsў
*transformer_block/layer_normalization/FillFill8transformer_block/layer_normalization/Fill/dims:output:04transformer_block/layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2,
*transformer_block/layer_normalization/FillЃ
-transformer_block/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2/
-transformer_block/layer_normalization/Const_1Э
1transformer_block/layer_normalization/Fill_1/dimsPack/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:23
1transformer_block/layer_normalization/Fill_1/dims
,transformer_block/layer_normalization/Fill_1Fill:transformer_block/layer_normalization/Fill_1/dims:output:06transformer_block/layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,transformer_block/layer_normalization/Fill_1Ё
-transformer_block/layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2/
-transformer_block/layer_normalization/Const_2Ё
-transformer_block/layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2/
-transformer_block/layer_normalization/Const_3Э
6transformer_block/layer_normalization/FusedBatchNormV3FusedBatchNormV36transformer_block/layer_normalization/Reshape:output:03transformer_block/layer_normalization/Fill:output:05transformer_block/layer_normalization/Fill_1:output:06transformer_block/layer_normalization/Const_2:output:06transformer_block/layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:28
6transformer_block/layer_normalization/FusedBatchNormV3
/transformer_block/layer_normalization/Reshape_1Reshape:transformer_block/layer_normalization/FusedBatchNormV3:y:04transformer_block/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х21
/transformer_block/layer_normalization/Reshape_1љ
:transformer_block/layer_normalization/mul_3/ReadVariableOpReadVariableOpCtransformer_block_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02<
:transformer_block/layer_normalization/mul_3/ReadVariableOp
+transformer_block/layer_normalization/mul_3Mul8transformer_block/layer_normalization/Reshape_1:output:0Btransformer_block/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+transformer_block/layer_normalization/mul_3ѓ
8transformer_block/layer_normalization/add/ReadVariableOpReadVariableOpAtransformer_block_layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8transformer_block/layer_normalization/add/ReadVariableOp
)transformer_block/layer_normalization/addAddV2/transformer_block/layer_normalization/mul_3:z:0@transformer_block/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)transformer_block/layer_normalization/add
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02?
=transformer_block/sequential/dense_4/Tensordot/ReadVariableOpД
3transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3transformer_block/sequential/dense_4/Tensordot/axesЛ
3transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3transformer_block/sequential/dense_4/Tensordot/freeЩ
4transformer_block/sequential/dense_4/Tensordot/ShapeShape-transformer_block/layer_normalization/add:z:0*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_4/Tensordot/ShapeО
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_4/Tensordot/GatherV2/axis
7transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/free:output:0Etransformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7transformer_block/sequential/dense_4/Tensordot/GatherV2Т
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_4/Tensordot/Shape:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Gtransformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/sequential/dense_4/Tensordot/GatherV2_1Ж
4transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/sequential/dense_4/Tensordot/Const
3transformer_block/sequential/dense_4/Tensordot/ProdProd@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3transformer_block/sequential/dense_4/Tensordot/ProdК
6transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/sequential/dense_4/Tensordot/Const_1
5transformer_block/sequential/dense_4/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5transformer_block/sequential/dense_4/Tensordot/Prod_1К
:transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block/sequential/dense_4/Tensordot/concat/axisщ
5transformer_block/sequential/dense_4/Tensordot/concatConcatV2<transformer_block/sequential/dense_4/Tensordot/free:output:0<transformer_block/sequential/dense_4/Tensordot/axes:output:0Ctransformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_4/Tensordot/concat 
4transformer_block/sequential/dense_4/Tensordot/stackPack<transformer_block/sequential/dense_4/Tensordot/Prod:output:0>transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_4/Tensordot/stackЇ
8transformer_block/sequential/dense_4/Tensordot/transpose	Transpose-transformer_block/layer_normalization/add:z:0>transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2:
8transformer_block/sequential/dense_4/Tensordot/transposeГ
6transformer_block/sequential/dense_4/Tensordot/ReshapeReshape<transformer_block/sequential/dense_4/Tensordot/transpose:y:0=transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6transformer_block/sequential/dense_4/Tensordot/ReshapeГ
5transformer_block/sequential/dense_4/Tensordot/MatMulMatMul?transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ27
5transformer_block/sequential/dense_4/Tensordot/MatMulЛ
6transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:28
6transformer_block/sequential/dense_4/Tensordot/Const_2О
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_4/Tensordot/concat_1/axisі
7transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/sequential/dense_4/Tensordot/concat_1Ѕ
.transformer_block/sequential/dense_4/TensordotReshape?transformer_block/sequential/dense_4/Tensordot/MatMul:product:0@transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ220
.transformer_block/sequential/dense_4/Tensordotќ
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02=
;transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp
,transformer_block/sequential/dense_4/BiasAddBiasAdd7transformer_block/sequential/dense_4/Tensordot:output:0Ctransformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22.
,transformer_block/sequential/dense_4/BiasAddм
1transformer_block/sequential/dense_4/p_re_lu/ReluRelu5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ223
1transformer_block/sequential/dense_4/p_re_lu/Relu
;transformer_block/sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02=
;transformer_block/sequential/dense_4/p_re_lu/ReadVariableOpк
0transformer_block/sequential/dense_4/p_re_lu/NegNegCtransformer_block/sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	222
0transformer_block/sequential/dense_4/p_re_lu/Negн
2transformer_block/sequential/dense_4/p_re_lu/Neg_1Neg5transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ224
2transformer_block/sequential/dense_4/p_re_lu/Neg_1с
3transformer_block/sequential/dense_4/p_re_lu/Relu_1Relu6transformer_block/sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ225
3transformer_block/sequential/dense_4/p_re_lu/Relu_1
0transformer_block/sequential/dense_4/p_re_lu/mulMul4transformer_block/sequential/dense_4/p_re_lu/Neg:y:0Atransformer_block/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ222
0transformer_block/sequential/dense_4/p_re_lu/mul
0transformer_block/sequential/dense_4/p_re_lu/addAddV2?transformer_block/sequential/dense_4/p_re_lu/Relu:activations:04transformer_block/sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ222
0transformer_block/sequential/dense_4/p_re_lu/add
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpFtransformer_block_sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02?
=transformer_block/sequential/dense_5/Tensordot/ReadVariableOpД
3transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:25
3transformer_block/sequential/dense_5/Tensordot/axesЛ
3transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       25
3transformer_block/sequential/dense_5/Tensordot/freeа
4transformer_block/sequential/dense_5/Tensordot/ShapeShape4transformer_block/sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_5/Tensordot/ShapeО
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_5/Tensordot/GatherV2/axis
7transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/free:output:0Etransformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:29
7transformer_block/sequential/dense_5/Tensordot/GatherV2Т
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2@
>transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2=transformer_block/sequential/dense_5/Tensordot/Shape:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Gtransformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2;
9transformer_block/sequential/dense_5/Tensordot/GatherV2_1Ж
4transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 26
4transformer_block/sequential/dense_5/Tensordot/Const
3transformer_block/sequential/dense_5/Tensordot/ProdProd@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0=transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 25
3transformer_block/sequential/dense_5/Tensordot/ProdК
6transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6transformer_block/sequential/dense_5/Tensordot/Const_1
5transformer_block/sequential/dense_5/Tensordot/Prod_1ProdBtransformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0?transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 27
5transformer_block/sequential/dense_5/Tensordot/Prod_1К
:transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2<
:transformer_block/sequential/dense_5/Tensordot/concat/axisщ
5transformer_block/sequential/dense_5/Tensordot/concatConcatV2<transformer_block/sequential/dense_5/Tensordot/free:output:0<transformer_block/sequential/dense_5/Tensordot/axes:output:0Ctransformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/sequential/dense_5/Tensordot/concat 
4transformer_block/sequential/dense_5/Tensordot/stackPack<transformer_block/sequential/dense_5/Tensordot/Prod:output:0>transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:26
4transformer_block/sequential/dense_5/Tensordot/stackЎ
8transformer_block/sequential/dense_5/Tensordot/transpose	Transpose4transformer_block/sequential/dense_4/p_re_lu/add:z:0>transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22:
8transformer_block/sequential/dense_5/Tensordot/transposeГ
6transformer_block/sequential/dense_5/Tensordot/ReshapeReshape<transformer_block/sequential/dense_5/Tensordot/transpose:y:0=transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ28
6transformer_block/sequential/dense_5/Tensordot/ReshapeГ
5transformer_block/sequential/dense_5/Tensordot/MatMulMatMul?transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Etransformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ27
5transformer_block/sequential/dense_5/Tensordot/MatMulЛ
6transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х28
6transformer_block/sequential/dense_5/Tensordot/Const_2О
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2>
<transformer_block/sequential/dense_5/Tensordot/concat_1/axisі
7transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2@transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0?transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Etransformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:29
7transformer_block/sequential/dense_5/Tensordot/concat_1Ѕ
.transformer_block/sequential/dense_5/TensordotReshape?transformer_block/sequential/dense_5/Tensordot/MatMul:product:0@transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х20
.transformer_block/sequential/dense_5/Tensordotќ
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpDtransformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02=
;transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp
,transformer_block/sequential/dense_5/BiasAddBiasAdd7transformer_block/sequential/dense_5/Tensordot:output:0Ctransformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2.
,transformer_block/sequential/dense_5/BiasAddЦ
$transformer_block/dropout_1/IdentityIdentity5transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2&
$transformer_block/dropout_1/Identityа
transformer_block/add_1AddV2-transformer_block/layer_normalization/add:z:0-transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
transformer_block/add_1Љ
-transformer_block/layer_normalization_1/ShapeShapetransformer_block/add_1:z:0*
T0*
_output_shapes
:2/
-transformer_block/layer_normalization_1/ShapeФ
;transformer_block/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;transformer_block/layer_normalization_1/strided_slice/stackШ
=transformer_block/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice/stack_1Ш
=transformer_block/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice/stack_2в
5transformer_block/layer_normalization_1/strided_sliceStridedSlice6transformer_block/layer_normalization_1/Shape:output:0Dtransformer_block/layer_normalization_1/strided_slice/stack:output:0Ftransformer_block/layer_normalization_1/strided_slice/stack_1:output:0Ftransformer_block/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5transformer_block/layer_normalization_1/strided_slice 
-transformer_block/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2/
-transformer_block/layer_normalization_1/mul/xњ
+transformer_block/layer_normalization_1/mulMul6transformer_block/layer_normalization_1/mul/x:output:0>transformer_block/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2-
+transformer_block/layer_normalization_1/mulШ
=transformer_block/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice_1/stackЬ
?transformer_block/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_1/stack_1Ь
?transformer_block/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_1/stack_2м
7transformer_block/layer_normalization_1/strided_slice_1StridedSlice6transformer_block/layer_normalization_1/Shape:output:0Ftransformer_block/layer_normalization_1/strided_slice_1/stack:output:0Htransformer_block/layer_normalization_1/strided_slice_1/stack_1:output:0Htransformer_block/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7transformer_block/layer_normalization_1/strided_slice_1љ
-transformer_block/layer_normalization_1/mul_1Mul/transformer_block/layer_normalization_1/mul:z:0@transformer_block/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2/
-transformer_block/layer_normalization_1/mul_1Ш
=transformer_block/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=transformer_block/layer_normalization_1/strided_slice_2/stackЬ
?transformer_block/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_2/stack_1Ь
?transformer_block/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?transformer_block/layer_normalization_1/strided_slice_2/stack_2м
7transformer_block/layer_normalization_1/strided_slice_2StridedSlice6transformer_block/layer_normalization_1/Shape:output:0Ftransformer_block/layer_normalization_1/strided_slice_2/stack:output:0Htransformer_block/layer_normalization_1/strided_slice_2/stack_1:output:0Htransformer_block/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7transformer_block/layer_normalization_1/strided_slice_2Є
/transformer_block/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :21
/transformer_block/layer_normalization_1/mul_2/x
-transformer_block/layer_normalization_1/mul_2Mul8transformer_block/layer_normalization_1/mul_2/x:output:0@transformer_block/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2/
-transformer_block/layer_normalization_1/mul_2Д
7transformer_block/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :29
7transformer_block/layer_normalization_1/Reshape/shape/0Д
7transformer_block/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :29
7transformer_block/layer_normalization_1/Reshape/shape/3
5transformer_block/layer_normalization_1/Reshape/shapePack@transformer_block/layer_normalization_1/Reshape/shape/0:output:01transformer_block/layer_normalization_1/mul_1:z:01transformer_block/layer_normalization_1/mul_2:z:0@transformer_block/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:27
5transformer_block/layer_normalization_1/Reshape/shape
/transformer_block/layer_normalization_1/ReshapeReshapetransformer_block/add_1:z:0>transformer_block/layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ21
/transformer_block/layer_normalization_1/ReshapeЃ
-transformer_block/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2/
-transformer_block/layer_normalization_1/ConstЯ
1transformer_block/layer_normalization_1/Fill/dimsPack1transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:23
1transformer_block/layer_normalization_1/Fill/dims
,transformer_block/layer_normalization_1/FillFill:transformer_block/layer_normalization_1/Fill/dims:output:06transformer_block/layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2.
,transformer_block/layer_normalization_1/FillЇ
/transformer_block/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    21
/transformer_block/layer_normalization_1/Const_1г
3transformer_block/layer_normalization_1/Fill_1/dimsPack1transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:25
3transformer_block/layer_normalization_1/Fill_1/dims
.transformer_block/layer_normalization_1/Fill_1Fill<transformer_block/layer_normalization_1/Fill_1/dims:output:08transformer_block/layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ20
.transformer_block/layer_normalization_1/Fill_1Ѕ
/transformer_block/layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 21
/transformer_block/layer_normalization_1/Const_2Ѕ
/transformer_block/layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 21
/transformer_block/layer_normalization_1/Const_3л
8transformer_block/layer_normalization_1/FusedBatchNormV3FusedBatchNormV38transformer_block/layer_normalization_1/Reshape:output:05transformer_block/layer_normalization_1/Fill:output:07transformer_block/layer_normalization_1/Fill_1:output:08transformer_block/layer_normalization_1/Const_2:output:08transformer_block/layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2:
8transformer_block/layer_normalization_1/FusedBatchNormV3
1transformer_block/layer_normalization_1/Reshape_1Reshape<transformer_block/layer_normalization_1/FusedBatchNormV3:y:06transformer_block/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х23
1transformer_block/layer_normalization_1/Reshape_1џ
<transformer_block/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpEtransformer_block_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02>
<transformer_block/layer_normalization_1/mul_3/ReadVariableOp
-transformer_block/layer_normalization_1/mul_3Mul:transformer_block/layer_normalization_1/Reshape_1:output:0Dtransformer_block/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2/
-transformer_block/layer_normalization_1/mul_3љ
:transformer_block/layer_normalization_1/add/ReadVariableOpReadVariableOpCtransformer_block_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02<
:transformer_block/layer_normalization_1/add/ReadVariableOp
+transformer_block/layer_normalization_1/addAddV21transformer_block/layer_normalization_1/mul_3:z:0Btransformer_block/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+transformer_block/layer_normalization_1/addЄ
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/global_average_pooling1d/Mean/reduction_indicesф
global_average_pooling1d/MeanMean/transformer_block/layer_normalization_1/add:z:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
global_average_pooling1d/Mean
dropout_2/IdentityIdentity&global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout_2/IdentityЇ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
dense_6/MatMul/ReadVariableOpЁ
dense_6/MatMulMatMuldropout_2/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_6/MatMulЅ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpЂ
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_6/BiasAddu
p_re_lu_1/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Relu
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*
_output_shapes	
:*
dtype02
p_re_lu_1/ReadVariableOpm
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
p_re_lu_1/Negv
p_re_lu_1/Neg_1Negdense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Neg_1t
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/Relu_1
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/mul
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
p_re_lu_1/addz
dropout_3/IdentityIdentityp_re_lu_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout_3/IdentityІ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldropout_3/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/MatMulЄ
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_7/BiasAdd/ReadVariableOpЁ
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
dense_7/Softmaxm
IdentityIdentitydense_7/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х:::::::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs

­
B__inference_dense_5_layer_call_and_return_conditional_losses_67328

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ2:::T P
,
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs
ю
m
'__inference_p_re_lu_layer_call_fn_64089

inputs
unknown
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_p_re_lu_layer_call_and_return_conditional_losses_640812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:22
StatefulPartitionedCallStatefulPartitionedCall:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
г
Ќ
*__inference_sequential_layer_call_fn_67234

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_642332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
к
|
'__inference_dense_7_layer_call_fn_67091

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_651642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_67061

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

Й
,__inference_functional_1_layer_call_fn_65445
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_653982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1


E__inference_sequential_layer_call_and_return_conditional_losses_64214
dense_4_input
dense_4_64201
dense_4_64203
dense_4_64205
dense_5_64208
dense_5_64210
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЉ
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_64201dense_4_64203dense_4_64205*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_641312!
dense_4/StatefulPartitionedCallГ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_64208dense_5_64210*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_641812!
dense_5/StatefulPartitionedCallХ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ2Х
'
_user_specified_namedense_4_input
*

G__inference_functional_1_layer_call_and_return_conditional_losses_65294

inputs
transformer_block_65242
transformer_block_65244
transformer_block_65246
transformer_block_65248
transformer_block_65250
transformer_block_65252
transformer_block_65254
transformer_block_65256
transformer_block_65258
transformer_block_65260
transformer_block_65262
transformer_block_65264
transformer_block_65266
transformer_block_65268
transformer_block_65270
transformer_block_65272
transformer_block_65274
dense_6_65279
dense_6_65281
p_re_lu_1_65284
dense_7_65288
dense_7_65290
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ!dropout_2/StatefulPartitionedCallЂ!dropout_3/StatefulPartitionedCallЂ!p_re_lu_1/StatefulPartitionedCallЂ)transformer_block/StatefulPartitionedCallи
)transformer_block/StatefulPartitionedCallStatefulPartitionedCallinputstransformer_block_65242transformer_block_65244transformer_block_65246transformer_block_65248transformer_block_65250transformer_block_65252transformer_block_65254transformer_block_65256transformer_block_65258transformer_block_65260transformer_block_65262transformer_block_65264transformer_block_65266transformer_block_65268transformer_block_65270transformer_block_65272transformer_block_65274*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_646362+
)transformer_block/StatefulPartitionedCallА
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_650572*
(global_average_pooling1d/PartitionedCall
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650762#
!dropout_2/StatefulPartitionedCallБ
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0dense_6_65279dense_6_65281*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_651042!
dense_6/StatefulPartitionedCallІ
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0p_re_lu_1_65284*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_643092#
!p_re_lu_1/StatefulPartitionedCallЗ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651352#
!dropout_3/StatefulPartitionedCallА
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0dense_7_65288dense_7_65290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_651642!
dense_7/StatefulPartitionedCallи
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
н
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_65057

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ2Х:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
'
Ш
G__inference_functional_1_layer_call_and_return_conditional_losses_65398

inputs
transformer_block_65346
transformer_block_65348
transformer_block_65350
transformer_block_65352
transformer_block_65354
transformer_block_65356
transformer_block_65358
transformer_block_65360
transformer_block_65362
transformer_block_65364
transformer_block_65366
transformer_block_65368
transformer_block_65370
transformer_block_65372
transformer_block_65374
transformer_block_65376
transformer_block_65378
dense_6_65383
dense_6_65385
p_re_lu_1_65388
dense_7_65392
dense_7_65394
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂ!p_re_lu_1/StatefulPartitionedCallЂ)transformer_block/StatefulPartitionedCallи
)transformer_block/StatefulPartitionedCallStatefulPartitionedCallinputstransformer_block_65346transformer_block_65348transformer_block_65350transformer_block_65352transformer_block_65354transformer_block_65356transformer_block_65358transformer_block_65360transformer_block_65362transformer_block_65364transformer_block_65366transformer_block_65368transformer_block_65370transformer_block_65372transformer_block_65374transformer_block_65376transformer_block_65378*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_649372+
)transformer_block/StatefulPartitionedCallА
(global_average_pooling1d/PartitionedCallPartitionedCall2transformer_block/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_650572*
(global_average_pooling1d/PartitionedCall
dropout_2/PartitionedCallPartitionedCall1global_average_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650812
dropout_2/PartitionedCallЉ
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0dense_6_65383dense_6_65385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_651042!
dense_6/StatefulPartitionedCallІ
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0p_re_lu_1_65388*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_643092#
!p_re_lu_1/StatefulPartitionedCallћ
dropout_3/PartitionedCallPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_3_layer_call_and_return_conditional_losses_651402
dropout_3/PartitionedCallЈ
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0dense_7_65392dense_7_65394*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_651642!
dense_7/StatefulPartitionedCall
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0 ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall*^transformer_block/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2V
)transformer_block/StatefulPartitionedCall)transformer_block/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
э

э
1__inference_transformer_block_layer_call_fn_66976

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15
identityЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_transformer_block_layer_call_and_return_conditional_losses_649372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
В
Њ
B__inference_dense_7_layer_call_and_return_conditional_losses_67082

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
	
~
B__inference_p_re_lu_layer_call_and_return_conditional_losses_64081

inputs
readvariableop_resource
identityd
ReluReluinputs*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Reluy
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	2*
dtype02
ReadVariableOpS
NegNegReadVariableOp:value:0*
T0*
_output_shapes
:	22
Nege
Neg_1Neginputs*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Neg_1k
Relu_1Relu	Neg_1:y:0*
T0*=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu_1g
mulMulNeg:y:0Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22
mulg
addAddV2Relu:activations:0mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22

Identity"
identityIdentity:output:0*@
_input_shapes/
-:'џџџџџџџџџџџџџџџџџџџџџџџџџџџ::e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы

E__inference_sequential_layer_call_and_return_conditional_losses_64264

inputs
dense_4_64251
dense_4_64253
dense_4_64255
dense_5_64258
dense_5_64260
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_64251dense_4_64253dense_4_64255*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_641312!
dense_4/StatefulPartitionedCallГ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_64258dense_5_64260*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_641812!
dense_5/StatefulPartitionedCallХ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs

c
D__inference_dropout_3_layer_call_and_return_conditional_losses_67056

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

c
D__inference_dropout_2_layer_call_and_return_conditional_losses_65076

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЕ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yП
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:џџџџџџџџџХ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџХ:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_64293

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ќы
	
L__inference_transformer_block_layer_call_and_return_conditional_losses_64636

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource5
1layer_normalization_mul_3_readvariableop_resource3
/layer_normalization_add_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource6
2sequential_dense_4_p_re_lu_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource7
3layer_normalization_1_mul_3_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource
identityx
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:2!
multi_head_self_attention/ShapeЈ
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-multi_head_self_attention/strided_slice/stackЌ
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_1Ќ
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_2ў
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'multi_head_self_attention/strided_sliceј
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02:
8multi_head_self_attention/dense/Tensordot/ReadVariableOpЊ
.multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_self_attention/dense/Tensordot/axesБ
.multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_self_attention/dense/Tensordot/free
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/ShapeД
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/GatherV2/axisё
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/GatherV2И
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisї
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense/Tensordot/GatherV2_1Ќ
/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention/dense/Tensordot/Const
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_self_attention/dense/Tensordot/ProdА
1multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense/Tensordot/Const_1
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense/Tensordot/Prod_1А
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_self_attention/dense/Tensordot/concat/axisа
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_self_attention/dense/Tensordot/concat
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/stackё
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х25
3multi_head_self_attention/dense/Tensordot/transpose
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1multi_head_self_attention/dense/Tensordot/Reshape
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ22
0multi_head_self_attention/dense/Tensordot/MatMulБ
1multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х23
1multi_head_self_attention/dense/Tensordot/Const_2Д
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/concat_1/axisн
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/concat_1
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense/Tensordotэ
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype028
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2)
'multi_head_self_attention/dense/BiasAddў
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_1/Tensordot/axesЕ
0multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_1/Tensordot/free
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/ShapeИ
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/GatherV2М
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_1/Tensordot/Const
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_1/Tensordot/ProdД
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_1/Tensordot/Const_1
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_1/Tensordot/Prod_1Д
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_1/Tensordot/concat/axisк
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_1/Tensordot/concat
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/stackї
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_1/Tensordot/transposeЇ
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_1/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_1/Tensordot/MatMulЕ
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_1/Tensordot/Const_2И
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/concat_1
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_1/Tensordotѓ
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_1/BiasAddў
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_2/Tensordot/axesЕ
0multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_2/Tensordot/free
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/ShapeИ
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/GatherV2М
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_2/Tensordot/Const
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_2/Tensordot/ProdД
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_2/Tensordot/Const_1
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_2/Tensordot/Prod_1Д
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_2/Tensordot/concat/axisк
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_2/Tensordot/concat
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/stackї
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_2/Tensordot/transposeЇ
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_2/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_2/Tensordot/MatMulЕ
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_2/Tensordot/Const_2И
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/concat_1
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_2/Tensordotѓ
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_2/BiasAddЁ
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)multi_head_self_attention/Reshape/shape/1
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/2
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/3ж
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_self_attention/Reshape/shapeј
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Reshape­
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(multi_head_self_attention/transpose/permљ
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/transposeЅ
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_1/shape/1
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/2
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/3р
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_1/shape
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_1Б
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_1/perm
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_1Ѕ
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_2/shape/1
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/2
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/3р
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_2/shape
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_2Б
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_2/perm
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_2
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(2"
 multi_head_self_attention/MatMul
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:2#
!multi_head_self_attention/Shape_1Е
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ21
/multi_head_self_attention/strided_slice_1/stackА
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/strided_slice_1/stack_1А
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention/strided_slice_1/stack_2
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention/strided_slice_1Ќ
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
multi_head_self_attention/Cast
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 2 
multi_head_self_attention/Sqrtь
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/truedivФ
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Softmaxє
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2$
"multi_head_self_attention/MatMul_1Б
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_3/perm
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_3Ѕ
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_3/shape/1
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2-
+multi_head_self_attention/Reshape_3/shape/2Њ
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_3/shapeє
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2%
#multi_head_self_attention/Reshape_3ў
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_3/Tensordot/axesЕ
0multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_3/Tensordot/freeТ
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/ShapeИ
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/GatherV2М
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_3/Tensordot/Const
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_3/Tensordot/ProdД
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_3/Tensordot/Const_1
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_3/Tensordot/Prod_1Д
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_3/Tensordot/concat/axisк
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_3/Tensordot/concat
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/stackІ
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5multi_head_self_attention/dense_3/Tensordot/transposeЇ
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_3/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_3/Tensordot/MatMulЕ
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_3/Tensordot/Const_2И
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/concat_1Ђ
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2-
+multi_head_self_attention/dense_3/Tensordotѓ
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2+
)multi_head_self_attention/dense_3/BiasAdds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstХ
dropout/dropout/MulMul2multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/Mul
dropout/dropout/ShapeShape2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeк
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yь
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/GreaterEqualЅ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/CastЈ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/Mul_1m
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
addm
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:2
layer_normalization/Shape
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack 
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1 
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2к
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/xЊ
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul 
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stackЄ
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1Є
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2ф
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1Љ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1 
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_2/stackЄ
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_1Є
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_2ф
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_2|
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_2/xВ
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_2
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shapeН
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization/Const
layer_normalization/Fill/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dimsЖ
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1
layer_normalization/Fill_1/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dimsО
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3Я
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2&
$layer_normalization/FusedBatchNormV3Ю
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/Reshape_1У
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization/mul_3/ReadVariableOpЮ
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/mul_3Н
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02(
&layer_normalization/add/ReadVariableOpС
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/addб
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_4/Tensordot/ReadVariableOp
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_4/Tensordot/axes
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_4/Tensordot/free
"sequential/dense_4/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/Shape
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/GatherV2/axisА
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/GatherV2
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_4/Tensordot/GatherV2_1/axisЖ
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_4/Tensordot/GatherV2_1
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_4/Tensordot/ConstЬ
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_4/Tensordot/Prod
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_4/Tensordot/Const_1д
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_4/Tensordot/Prod_1
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_4/Tensordot/concat/axis
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_4/Tensordot/concatи
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/stackп
&sequential/dense_4/Tensordot/transpose	Transposelayer_normalization/add:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2(
&sequential/dense_4/Tensordot/transposeы
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_4/Tensordot/Reshapeы
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#sequential/dense_4/Tensordot/MatMul
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/dense_4/Tensordot/Const_2
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/concat_1/axis
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/concat_1н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/TensordotЦ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)sequential/dense_4/BiasAdd/ReadVariableOpд
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/BiasAddІ
sequential/dense_4/p_re_lu/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22!
sequential/dense_4/p_re_lu/ReluЪ
)sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOp2sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02+
)sequential/dense_4/p_re_lu/ReadVariableOpЄ
sequential/dense_4/p_re_lu/NegNeg1sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22 
sequential/dense_4/p_re_lu/NegЇ
 sequential/dense_4/p_re_lu/Neg_1Neg#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22"
 sequential/dense_4/p_re_lu/Neg_1Ћ
!sequential/dense_4/p_re_lu/Relu_1Relu$sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22#
!sequential/dense_4/p_re_lu/Relu_1г
sequential/dense_4/p_re_lu/mulMul"sequential/dense_4/p_re_lu/Neg:y:0/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/mulг
sequential/dense_4/p_re_lu/addAddV2-sequential/dense_4/p_re_lu/Relu:activations:0"sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/addб
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_5/Tensordot/ReadVariableOp
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_5/Tensordot/axes
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_5/Tensordot/free
"sequential/dense_5/Tensordot/ShapeShape"sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/Shape
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/GatherV2/axisА
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/GatherV2
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_5/Tensordot/GatherV2_1/axisЖ
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_5/Tensordot/GatherV2_1
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_5/Tensordot/ConstЬ
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_5/Tensordot/Prod
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_5/Tensordot/Const_1д
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_5/Tensordot/Prod_1
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_5/Tensordot/concat/axis
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_5/Tensordot/concatи
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/stackц
&sequential/dense_5/Tensordot/transpose	Transpose"sequential/dense_4/p_re_lu/add:z:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22(
&sequential/dense_5/Tensordot/transposeы
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_5/Tensordot/Reshapeы
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2%
#sequential/dense_5/Tensordot/MatMul
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2&
$sequential/dense_5/Tensordot/Const_2
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/concat_1/axis
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/concat_1н
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/TensordotЦ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02+
)sequential/dense_5/BiasAdd/ReadVariableOpд
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstГ
dropout_1/dropout/MulMul#sequential/dense_5/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeз
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yы
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2 
dropout_1/dropout/GreaterEqualЂ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/CastЇ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/Mul_1
add_1AddV2layer_normalization/add:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
add_1s
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape 
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stackЄ
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1Є
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2ц
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/xВ
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mulЄ
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stackЈ
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1Ј
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2№
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1Б
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1Є
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_2/stackЈ
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_1Ј
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_2№
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_2
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_2/xК
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_2
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3Ђ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shapeХ
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization_1/Const
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dimsО
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dimsЦ
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill_1
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3н
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2(
&layer_normalization_1/FusedBatchNormV3ж
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2!
layer_normalization_1/Reshape_1Щ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02,
*layer_normalization_1/mul_3/ReadVariableOpж
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/mul_3У
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization_1/add/ReadVariableOpЩ
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/addv
IdentityIdentitylayer_normalization_1/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
хи
	
L__inference_transformer_block_layer_call_and_return_conditional_losses_66898

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource5
1layer_normalization_mul_3_readvariableop_resource3
/layer_normalization_add_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource6
2sequential_dense_4_p_re_lu_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource7
3layer_normalization_1_mul_3_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource
identityx
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:2!
multi_head_self_attention/ShapeЈ
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-multi_head_self_attention/strided_slice/stackЌ
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_1Ќ
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_2ў
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'multi_head_self_attention/strided_sliceј
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02:
8multi_head_self_attention/dense/Tensordot/ReadVariableOpЊ
.multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_self_attention/dense/Tensordot/axesБ
.multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_self_attention/dense/Tensordot/free
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/ShapeД
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/GatherV2/axisё
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/GatherV2И
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisї
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense/Tensordot/GatherV2_1Ќ
/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention/dense/Tensordot/Const
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_self_attention/dense/Tensordot/ProdА
1multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense/Tensordot/Const_1
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense/Tensordot/Prod_1А
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_self_attention/dense/Tensordot/concat/axisа
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_self_attention/dense/Tensordot/concat
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/stackё
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х25
3multi_head_self_attention/dense/Tensordot/transpose
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1multi_head_self_attention/dense/Tensordot/Reshape
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ22
0multi_head_self_attention/dense/Tensordot/MatMulБ
1multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х23
1multi_head_self_attention/dense/Tensordot/Const_2Д
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/concat_1/axisн
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/concat_1
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense/Tensordotэ
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype028
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2)
'multi_head_self_attention/dense/BiasAddў
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_1/Tensordot/axesЕ
0multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_1/Tensordot/free
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/ShapeИ
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/GatherV2М
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_1/Tensordot/Const
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_1/Tensordot/ProdД
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_1/Tensordot/Const_1
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_1/Tensordot/Prod_1Д
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_1/Tensordot/concat/axisк
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_1/Tensordot/concat
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/stackї
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_1/Tensordot/transposeЇ
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_1/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_1/Tensordot/MatMulЕ
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_1/Tensordot/Const_2И
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/concat_1
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_1/Tensordotѓ
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_1/BiasAddў
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_2/Tensordot/axesЕ
0multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_2/Tensordot/free
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/ShapeИ
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/GatherV2М
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_2/Tensordot/Const
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_2/Tensordot/ProdД
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_2/Tensordot/Const_1
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_2/Tensordot/Prod_1Д
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_2/Tensordot/concat/axisк
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_2/Tensordot/concat
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/stackї
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_2/Tensordot/transposeЇ
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_2/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_2/Tensordot/MatMulЕ
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_2/Tensordot/Const_2И
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/concat_1
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_2/Tensordotѓ
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_2/BiasAddЁ
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)multi_head_self_attention/Reshape/shape/1
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/2
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/3ж
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_self_attention/Reshape/shapeј
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Reshape­
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(multi_head_self_attention/transpose/permљ
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/transposeЅ
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_1/shape/1
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/2
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/3р
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_1/shape
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_1Б
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_1/perm
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_1Ѕ
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_2/shape/1
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/2
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/3р
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_2/shape
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_2Б
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_2/perm
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_2
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(2"
 multi_head_self_attention/MatMul
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:2#
!multi_head_self_attention/Shape_1Е
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ21
/multi_head_self_attention/strided_slice_1/stackА
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/strided_slice_1/stack_1А
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention/strided_slice_1/stack_2
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention/strided_slice_1Ќ
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
multi_head_self_attention/Cast
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 2 
multi_head_self_attention/Sqrtь
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/truedivФ
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Softmaxє
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2$
"multi_head_self_attention/MatMul_1Б
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_3/perm
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_3Ѕ
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_3/shape/1
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2-
+multi_head_self_attention/Reshape_3/shape/2Њ
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_3/shapeє
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2%
#multi_head_self_attention/Reshape_3ў
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_3/Tensordot/axesЕ
0multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_3/Tensordot/freeТ
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/ShapeИ
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/GatherV2М
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_3/Tensordot/Const
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_3/Tensordot/ProdД
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_3/Tensordot/Const_1
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_3/Tensordot/Prod_1Д
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_3/Tensordot/concat/axisк
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_3/Tensordot/concat
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/stackІ
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5multi_head_self_attention/dense_3/Tensordot/transposeЇ
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_3/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_3/Tensordot/MatMulЕ
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_3/Tensordot/Const_2И
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/concat_1Ђ
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2-
+multi_head_self_attention/dense_3/Tensordotѓ
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2+
)multi_head_self_attention/dense_3/BiasAddЄ
dropout/IdentityIdentity2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/Identitym
addAddV2inputsdropout/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
addm
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:2
layer_normalization/Shape
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack 
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1 
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2к
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/xЊ
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul 
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stackЄ
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1Є
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2ф
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1Љ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1 
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_2/stackЄ
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_1Є
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_2ф
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_2|
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_2/xВ
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_2
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shapeН
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization/Const
layer_normalization/Fill/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dimsЖ
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1
layer_normalization/Fill_1/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dimsО
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3Я
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2&
$layer_normalization/FusedBatchNormV3Ю
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/Reshape_1У
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization/mul_3/ReadVariableOpЮ
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/mul_3Н
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02(
&layer_normalization/add/ReadVariableOpС
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/addб
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_4/Tensordot/ReadVariableOp
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_4/Tensordot/axes
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_4/Tensordot/free
"sequential/dense_4/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/Shape
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/GatherV2/axisА
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/GatherV2
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_4/Tensordot/GatherV2_1/axisЖ
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_4/Tensordot/GatherV2_1
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_4/Tensordot/ConstЬ
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_4/Tensordot/Prod
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_4/Tensordot/Const_1д
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_4/Tensordot/Prod_1
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_4/Tensordot/concat/axis
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_4/Tensordot/concatи
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/stackп
&sequential/dense_4/Tensordot/transpose	Transposelayer_normalization/add:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2(
&sequential/dense_4/Tensordot/transposeы
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_4/Tensordot/Reshapeы
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#sequential/dense_4/Tensordot/MatMul
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/dense_4/Tensordot/Const_2
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/concat_1/axis
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/concat_1н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/TensordotЦ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)sequential/dense_4/BiasAdd/ReadVariableOpд
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/BiasAddІ
sequential/dense_4/p_re_lu/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22!
sequential/dense_4/p_re_lu/ReluЪ
)sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOp2sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02+
)sequential/dense_4/p_re_lu/ReadVariableOpЄ
sequential/dense_4/p_re_lu/NegNeg1sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22 
sequential/dense_4/p_re_lu/NegЇ
 sequential/dense_4/p_re_lu/Neg_1Neg#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22"
 sequential/dense_4/p_re_lu/Neg_1Ћ
!sequential/dense_4/p_re_lu/Relu_1Relu$sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22#
!sequential/dense_4/p_re_lu/Relu_1г
sequential/dense_4/p_re_lu/mulMul"sequential/dense_4/p_re_lu/Neg:y:0/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/mulг
sequential/dense_4/p_re_lu/addAddV2-sequential/dense_4/p_re_lu/Relu:activations:0"sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/addб
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_5/Tensordot/ReadVariableOp
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_5/Tensordot/axes
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_5/Tensordot/free
"sequential/dense_5/Tensordot/ShapeShape"sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/Shape
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/GatherV2/axisА
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/GatherV2
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_5/Tensordot/GatherV2_1/axisЖ
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_5/Tensordot/GatherV2_1
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_5/Tensordot/ConstЬ
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_5/Tensordot/Prod
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_5/Tensordot/Const_1д
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_5/Tensordot/Prod_1
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_5/Tensordot/concat/axis
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_5/Tensordot/concatи
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/stackц
&sequential/dense_5/Tensordot/transpose	Transpose"sequential/dense_4/p_re_lu/add:z:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22(
&sequential/dense_5/Tensordot/transposeы
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_5/Tensordot/Reshapeы
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2%
#sequential/dense_5/Tensordot/MatMul
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2&
$sequential/dense_5/Tensordot/Const_2
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/concat_1/axis
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/concat_1н
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/TensordotЦ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02+
)sequential/dense_5/BiasAdd/ReadVariableOpд
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/BiasAdd
dropout_1/IdentityIdentity#sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/Identity
add_1AddV2layer_normalization/add:z:0dropout_1/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
add_1s
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape 
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stackЄ
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1Є
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2ц
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/xВ
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mulЄ
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stackЈ
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1Ј
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2№
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1Б
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1Є
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_2/stackЈ
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_1Ј
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_2№
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_2
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_2/xК
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_2
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3Ђ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shapeХ
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization_1/Const
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dimsО
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dimsЦ
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill_1
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3н
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2(
&layer_normalization_1/FusedBatchNormV3ж
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2!
layer_normalization_1/Reshape_1Щ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02,
*layer_normalization_1/mul_3/ReadVariableOpж
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/mul_3У
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization_1/add/ReadVariableOpЩ
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/addv
IdentityIdentitylayer_normalization_1/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
Ы
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_67015

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџХ:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs
Ы
b
D__inference_dropout_3_layer_call_and_return_conditional_losses_65140

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы

E__inference_sequential_layer_call_and_return_conditional_losses_64233

inputs
dense_4_64220
dense_4_64222
dense_4_64224
dense_5_64227
dense_5_64229
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЂ
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_64220dense_4_64222dense_4_64224*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_641312!
dense_4/StatefulPartitionedCallГ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_64227dense_5_64229*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_641812!
dense_5/StatefulPartitionedCallХ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
г
Ќ
*__inference_sequential_layer_call_fn_67249

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identityЂStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*'
_read_only_resource_inputs	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_642642
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
ЁL
Щ
E__inference_sequential_layer_call_and_return_conditional_losses_67155

inputs-
)dense_4_tensordot_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource+
'dense_4_p_re_lu_readvariableop_resource-
)dense_5_tensordot_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityА
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/freeh
dense_4/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisљ
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axisџ
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1Ј
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisи
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concatЌ
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stackЉ
dense_4/Tensordot/transpose	Transposeinputs!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_4/Tensordot/transposeП
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_4/Tensordot/ReshapeП
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_4/Tensordot/MatMul
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axisх
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1Б
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/TensordotЅ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOpЈ
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/BiasAdd
dense_4/p_re_lu/ReluReludense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/ReluЉ
dense_4/p_re_lu/ReadVariableOpReadVariableOp'dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02 
dense_4/p_re_lu/ReadVariableOp
dense_4/p_re_lu/NegNeg&dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22
dense_4/p_re_lu/Neg
dense_4/p_re_lu/Neg_1Negdense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/Neg_1
dense_4/p_re_lu/Relu_1Reludense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/Relu_1Ї
dense_4/p_re_lu/mulMuldense_4/p_re_lu/Neg:y:0$dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/mulЇ
dense_4/p_re_lu/addAddV2"dense_4/p_re_lu/Relu:activations:0dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_4/p_re_lu/addА
 dense_5/Tensordot/ReadVariableOpReadVariableOp)dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02"
 dense_5/Tensordot/ReadVariableOpz
dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_5/Tensordot/axes
dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_5/Tensordot/freey
dense_5/Tensordot/ShapeShapedense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2
dense_5/Tensordot/Shape
dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/GatherV2/axisљ
dense_5/Tensordot/GatherV2GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/free:output:0(dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2
!dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_5/Tensordot/GatherV2_1/axisџ
dense_5/Tensordot/GatherV2_1GatherV2 dense_5/Tensordot/Shape:output:0dense_5/Tensordot/axes:output:0*dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_5/Tensordot/GatherV2_1|
dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const 
dense_5/Tensordot/ProdProd#dense_5/Tensordot/GatherV2:output:0 dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod
dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_5/Tensordot/Const_1Ј
dense_5/Tensordot/Prod_1Prod%dense_5/Tensordot/GatherV2_1:output:0"dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_5/Tensordot/Prod_1
dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_5/Tensordot/concat/axisи
dense_5/Tensordot/concatConcatV2dense_5/Tensordot/free:output:0dense_5/Tensordot/axes:output:0&dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concatЌ
dense_5/Tensordot/stackPackdense_5/Tensordot/Prod:output:0!dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/stackК
dense_5/Tensordot/transpose	Transposedense_4/p_re_lu/add:z:0!dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
dense_5/Tensordot/transposeП
dense_5/Tensordot/ReshapeReshapedense_5/Tensordot/transpose:y:0 dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
dense_5/Tensordot/ReshapeП
dense_5/Tensordot/MatMulMatMul"dense_5/Tensordot/Reshape:output:0(dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
dense_5/Tensordot/MatMul
dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2
dense_5/Tensordot/Const_2
dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_5/Tensordot/concat_1/axisх
dense_5/Tensordot/concat_1ConcatV2#dense_5/Tensordot/GatherV2:output:0"dense_5/Tensordot/Const_2:output:0(dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_5/Tensordot/concat_1Б
dense_5/TensordotReshape"dense_5/Tensordot/MatMul:product:0#dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_5/TensordotЅ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02 
dense_5/BiasAdd/ReadVariableOpЈ
dense_5/BiasAddBiasAdddense_5/Tensordot:output:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dense_5/BiasAddq
IdentityIdentitydense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
ќы
	
L__inference_transformer_block_layer_call_and_return_conditional_losses_66597

inputsE
Amulti_head_self_attention_dense_tensordot_readvariableop_resourceC
?multi_head_self_attention_dense_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_1_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_1_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_2_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_2_biasadd_readvariableop_resourceG
Cmulti_head_self_attention_dense_3_tensordot_readvariableop_resourceE
Amulti_head_self_attention_dense_3_biasadd_readvariableop_resource5
1layer_normalization_mul_3_readvariableop_resource3
/layer_normalization_add_readvariableop_resource8
4sequential_dense_4_tensordot_readvariableop_resource6
2sequential_dense_4_biasadd_readvariableop_resource6
2sequential_dense_4_p_re_lu_readvariableop_resource8
4sequential_dense_5_tensordot_readvariableop_resource6
2sequential_dense_5_biasadd_readvariableop_resource7
3layer_normalization_1_mul_3_readvariableop_resource5
1layer_normalization_1_add_readvariableop_resource
identityx
multi_head_self_attention/ShapeShapeinputs*
T0*
_output_shapes
:2!
multi_head_self_attention/ShapeЈ
-multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-multi_head_self_attention/strided_slice/stackЌ
/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_1Ќ
/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/multi_head_self_attention/strided_slice/stack_2ў
'multi_head_self_attention/strided_sliceStridedSlice(multi_head_self_attention/Shape:output:06multi_head_self_attention/strided_slice/stack:output:08multi_head_self_attention/strided_slice/stack_1:output:08multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'multi_head_self_attention/strided_sliceј
8multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02:
8multi_head_self_attention/dense/Tensordot/ReadVariableOpЊ
.multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:20
.multi_head_self_attention/dense/Tensordot/axesБ
.multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       20
.multi_head_self_attention/dense/Tensordot/free
/multi_head_self_attention/dense/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/ShapeД
7multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/GatherV2/axisё
2multi_head_self_attention/dense/Tensordot/GatherV2GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/free:output:0@multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/GatherV2И
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense/Tensordot/GatherV2_1/axisї
4multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV28multi_head_self_attention/dense/Tensordot/Shape:output:07multi_head_self_attention/dense/Tensordot/axes:output:0Bmulti_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense/Tensordot/GatherV2_1Ќ
/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/multi_head_self_attention/dense/Tensordot/Const
.multi_head_self_attention/dense/Tensordot/ProdProd;multi_head_self_attention/dense/Tensordot/GatherV2:output:08multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 20
.multi_head_self_attention/dense/Tensordot/ProdА
1multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense/Tensordot/Const_1
0multi_head_self_attention/dense/Tensordot/Prod_1Prod=multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0:multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense/Tensordot/Prod_1А
5multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5multi_head_self_attention/dense/Tensordot/concat/axisа
0multi_head_self_attention/dense/Tensordot/concatConcatV27multi_head_self_attention/dense/Tensordot/free:output:07multi_head_self_attention/dense/Tensordot/axes:output:0>multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:22
0multi_head_self_attention/dense/Tensordot/concat
/multi_head_self_attention/dense/Tensordot/stackPack7multi_head_self_attention/dense/Tensordot/Prod:output:09multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:21
/multi_head_self_attention/dense/Tensordot/stackё
3multi_head_self_attention/dense/Tensordot/transpose	Transposeinputs9multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х25
3multi_head_self_attention/dense/Tensordot/transpose
1multi_head_self_attention/dense/Tensordot/ReshapeReshape7multi_head_self_attention/dense/Tensordot/transpose:y:08multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ23
1multi_head_self_attention/dense/Tensordot/Reshape
0multi_head_self_attention/dense/Tensordot/MatMulMatMul:multi_head_self_attention/dense/Tensordot/Reshape:output:0@multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ22
0multi_head_self_attention/dense/Tensordot/MatMulБ
1multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х23
1multi_head_self_attention/dense/Tensordot/Const_2Д
7multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense/Tensordot/concat_1/axisн
2multi_head_self_attention/dense/Tensordot/concat_1ConcatV2;multi_head_self_attention/dense/Tensordot/GatherV2:output:0:multi_head_self_attention/dense/Tensordot/Const_2:output:0@multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense/Tensordot/concat_1
)multi_head_self_attention/dense/TensordotReshape:multi_head_self_attention/dense/Tensordot/MatMul:product:0;multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense/Tensordotэ
6multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp?multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype028
6multi_head_self_attention/dense/BiasAdd/ReadVariableOp
'multi_head_self_attention/dense/BiasAddBiasAdd2multi_head_self_attention/dense/Tensordot:output:0>multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2)
'multi_head_self_attention/dense/BiasAddў
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_1/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_1/Tensordot/axesЕ
0multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_1/Tensordot/free
1multi_head_self_attention/dense_1/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/ShapeИ
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/free:output:0Bmulti_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/GatherV2М
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_1/Tensordot/Shape:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0Dmulti_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_1/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_1/Tensordot/Const
0multi_head_self_attention/dense_1/Tensordot/ProdProd=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_1/Tensordot/ProdД
3multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_1/Tensordot/Const_1
2multi_head_self_attention/dense_1/Tensordot/Prod_1Prod?multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_1/Tensordot/Prod_1Д
7multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_1/Tensordot/concat/axisк
2multi_head_self_attention/dense_1/Tensordot/concatConcatV29multi_head_self_attention/dense_1/Tensordot/free:output:09multi_head_self_attention/dense_1/Tensordot/axes:output:0@multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_1/Tensordot/concat
1multi_head_self_attention/dense_1/Tensordot/stackPack9multi_head_self_attention/dense_1/Tensordot/Prod:output:0;multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_1/Tensordot/stackї
5multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_1/Tensordot/transposeЇ
3multi_head_self_attention/dense_1/Tensordot/ReshapeReshape9multi_head_self_attention/dense_1/Tensordot/transpose:y:0:multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_1/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_1/Tensordot/MatMulMatMul<multi_head_self_attention/dense_1/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_1/Tensordot/MatMulЕ
3multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_1/Tensordot/Const_2И
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_1/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_1/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_1/Tensordot/concat_1
+multi_head_self_attention/dense_1/TensordotReshape<multi_head_self_attention/dense_1/Tensordot/MatMul:product:0=multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_1/Tensordotѓ
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_1/BiasAddBiasAdd4multi_head_self_attention/dense_1/Tensordot:output:0@multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_1/BiasAddў
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_2/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_2/Tensordot/axesЕ
0multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_2/Tensordot/free
1multi_head_self_attention/dense_2/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/ShapeИ
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/free:output:0Bmulti_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/GatherV2М
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_2/Tensordot/Shape:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0Dmulti_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_2/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_2/Tensordot/Const
0multi_head_self_attention/dense_2/Tensordot/ProdProd=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_2/Tensordot/ProdД
3multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_2/Tensordot/Const_1
2multi_head_self_attention/dense_2/Tensordot/Prod_1Prod?multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_2/Tensordot/Prod_1Д
7multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_2/Tensordot/concat/axisк
2multi_head_self_attention/dense_2/Tensordot/concatConcatV29multi_head_self_attention/dense_2/Tensordot/free:output:09multi_head_self_attention/dense_2/Tensordot/axes:output:0@multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_2/Tensordot/concat
1multi_head_self_attention/dense_2/Tensordot/stackPack9multi_head_self_attention/dense_2/Tensordot/Prod:output:0;multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_2/Tensordot/stackї
5multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinputs;multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х27
5multi_head_self_attention/dense_2/Tensordot/transposeЇ
3multi_head_self_attention/dense_2/Tensordot/ReshapeReshape9multi_head_self_attention/dense_2/Tensordot/transpose:y:0:multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_2/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_2/Tensordot/MatMulMatMul<multi_head_self_attention/dense_2/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_2/Tensordot/MatMulЕ
3multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_2/Tensordot/Const_2И
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_2/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_2/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_2/Tensordot/concat_1
+multi_head_self_attention/dense_2/TensordotReshape<multi_head_self_attention/dense_2/Tensordot/MatMul:product:0=multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2-
+multi_head_self_attention/dense_2/Tensordotѓ
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_2/BiasAddBiasAdd4multi_head_self_attention/dense_2/Tensordot:output:0@multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2+
)multi_head_self_attention/dense_2/BiasAddЁ
)multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2+
)multi_head_self_attention/Reshape/shape/1
)multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/2
)multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2+
)multi_head_self_attention/Reshape/shape/3ж
'multi_head_self_attention/Reshape/shapePack0multi_head_self_attention/strided_slice:output:02multi_head_self_attention/Reshape/shape/1:output:02multi_head_self_attention/Reshape/shape/2:output:02multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2)
'multi_head_self_attention/Reshape/shapeј
!multi_head_self_attention/ReshapeReshape0multi_head_self_attention/dense/BiasAdd:output:00multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Reshape­
(multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2*
(multi_head_self_attention/transpose/permљ
#multi_head_self_attention/transpose	Transpose*multi_head_self_attention/Reshape:output:01multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/transposeЅ
+multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_1/shape/1
+multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/2
+multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_1/shape/3р
)multi_head_self_attention/Reshape_1/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_1/shape/1:output:04multi_head_self_attention/Reshape_1/shape/2:output:04multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_1/shape
#multi_head_self_attention/Reshape_1Reshape2multi_head_self_attention/dense_1/BiasAdd:output:02multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_1Б
*multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_1/perm
%multi_head_self_attention/transpose_1	Transpose,multi_head_self_attention/Reshape_1:output:03multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_1Ѕ
+multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_2/shape/1
+multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/2
+multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+multi_head_self_attention/Reshape_2/shape/3р
)multi_head_self_attention/Reshape_2/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_2/shape/1:output:04multi_head_self_attention/Reshape_2/shape/2:output:04multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_2/shape
#multi_head_self_attention/Reshape_2Reshape2multi_head_self_attention/dense_2/BiasAdd:output:02multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2%
#multi_head_self_attention/Reshape_2Б
*multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_2/perm
%multi_head_self_attention/transpose_2	Transpose,multi_head_self_attention/Reshape_2:output:03multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_2
 multi_head_self_attention/MatMulBatchMatMulV2'multi_head_self_attention/transpose:y:0)multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(2"
 multi_head_self_attention/MatMul
!multi_head_self_attention/Shape_1Shape)multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:2#
!multi_head_self_attention/Shape_1Е
/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ21
/multi_head_self_attention/strided_slice_1/stackА
1multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/strided_slice_1/stack_1А
1multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1multi_head_self_attention/strided_slice_1/stack_2
)multi_head_self_attention/strided_slice_1StridedSlice*multi_head_self_attention/Shape_1:output:08multi_head_self_attention/strided_slice_1/stack:output:0:multi_head_self_attention/strided_slice_1/stack_1:output:0:multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)multi_head_self_attention/strided_slice_1Ќ
multi_head_self_attention/CastCast2multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
multi_head_self_attention/Cast
multi_head_self_attention/SqrtSqrt"multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 2 
multi_head_self_attention/Sqrtь
!multi_head_self_attention/truedivRealDiv)multi_head_self_attention/MatMul:output:0"multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/truedivФ
!multi_head_self_attention/SoftmaxSoftmax%multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2#
!multi_head_self_attention/Softmaxє
"multi_head_self_attention/MatMul_1BatchMatMulV2+multi_head_self_attention/Softmax:softmax:0)multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2$
"multi_head_self_attention/MatMul_1Б
*multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*multi_head_self_attention/transpose_3/perm
%multi_head_self_attention/transpose_3	Transpose+multi_head_self_attention/MatMul_1:output:03multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2'
%multi_head_self_attention/transpose_3Ѕ
+multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2-
+multi_head_self_attention/Reshape_3/shape/1
+multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2-
+multi_head_self_attention/Reshape_3/shape/2Њ
)multi_head_self_attention/Reshape_3/shapePack0multi_head_self_attention/strided_slice:output:04multi_head_self_attention/Reshape_3/shape/1:output:04multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)multi_head_self_attention/Reshape_3/shapeє
#multi_head_self_attention/Reshape_3Reshape)multi_head_self_attention/transpose_3:y:02multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2%
#multi_head_self_attention/Reshape_3ў
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpCmulti_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02<
:multi_head_self_attention/dense_3/Tensordot/ReadVariableOpЎ
0multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:22
0multi_head_self_attention/dense_3/Tensordot/axesЕ
0multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       22
0multi_head_self_attention/dense_3/Tensordot/freeТ
1multi_head_self_attention/dense_3/Tensordot/ShapeShape,multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/ShapeИ
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/GatherV2/axisћ
4multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/free:output:0Bmulti_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/GatherV2М
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2=
;multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2:multi_head_self_attention/dense_3/Tensordot/Shape:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0Dmulti_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:28
6multi_head_self_attention/dense_3/Tensordot/GatherV2_1А
1multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 23
1multi_head_self_attention/dense_3/Tensordot/Const
0multi_head_self_attention/dense_3/Tensordot/ProdProd=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0:multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 22
0multi_head_self_attention/dense_3/Tensordot/ProdД
3multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3multi_head_self_attention/dense_3/Tensordot/Const_1
2multi_head_self_attention/dense_3/Tensordot/Prod_1Prod?multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0<multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 24
2multi_head_self_attention/dense_3/Tensordot/Prod_1Д
7multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 29
7multi_head_self_attention/dense_3/Tensordot/concat/axisк
2multi_head_self_attention/dense_3/Tensordot/concatConcatV29multi_head_self_attention/dense_3/Tensordot/free:output:09multi_head_self_attention/dense_3/Tensordot/axes:output:0@multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:24
2multi_head_self_attention/dense_3/Tensordot/concat
1multi_head_self_attention/dense_3/Tensordot/stackPack9multi_head_self_attention/dense_3/Tensordot/Prod:output:0;multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:23
1multi_head_self_attention/dense_3/Tensordot/stackІ
5multi_head_self_attention/dense_3/Tensordot/transpose	Transpose,multi_head_self_attention/Reshape_3:output:0;multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ27
5multi_head_self_attention/dense_3/Tensordot/transposeЇ
3multi_head_self_attention/dense_3/Tensordot/ReshapeReshape9multi_head_self_attention/dense_3/Tensordot/transpose:y:0:multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ25
3multi_head_self_attention/dense_3/Tensordot/ReshapeЇ
2multi_head_self_attention/dense_3/Tensordot/MatMulMatMul<multi_head_self_attention/dense_3/Tensordot/Reshape:output:0Bmulti_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ24
2multi_head_self_attention/dense_3/Tensordot/MatMulЕ
3multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х25
3multi_head_self_attention/dense_3/Tensordot/Const_2И
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9multi_head_self_attention/dense_3/Tensordot/concat_1/axisч
4multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2=multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0<multi_head_self_attention/dense_3/Tensordot/Const_2:output:0Bmulti_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:26
4multi_head_self_attention/dense_3/Tensordot/concat_1Ђ
+multi_head_self_attention/dense_3/TensordotReshape<multi_head_self_attention/dense_3/Tensordot/MatMul:product:0=multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2-
+multi_head_self_attention/dense_3/Tensordotѓ
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOpAmulti_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02:
8multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp
)multi_head_self_attention/dense_3/BiasAddBiasAdd4multi_head_self_attention/dense_3/Tensordot:output:0@multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2+
)multi_head_self_attention/dense_3/BiasAdds
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/ConstХ
dropout/dropout/MulMul2multi_head_self_attention/dense_3/BiasAdd:output:0dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/Mul
dropout/dropout/ShapeShape2multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shapeк
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yь
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/GreaterEqualЅ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/CastЈ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2
dropout/dropout/Mul_1m
addAddV2inputsdropout/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
addm
layer_normalization/ShapeShapeadd:z:0*
T0*
_output_shapes
:2
layer_normalization/Shape
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'layer_normalization/strided_slice/stack 
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_1 
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice/stack_2к
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!layer_normalization/strided_slicex
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul/xЊ
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul 
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_1/stackЄ
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_1Є
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_1/stack_2ф
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_1Љ
layer_normalization/mul_1Mullayer_normalization/mul:z:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_1 
)layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)layer_normalization/strided_slice_2/stackЄ
+layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_1Є
+layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization/strided_slice_2/stack_2ф
#layer_normalization/strided_slice_2StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_2/stack:output:04layer_normalization/strided_slice_2/stack_1:output:04layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization/strided_slice_2|
layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization/mul_2/xВ
layer_normalization/mul_2Mul$layer_normalization/mul_2/x:output:0,layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization/mul_2
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/0
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2%
#layer_normalization/Reshape/shape/3
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul_1:z:0layer_normalization/mul_2:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2#
!layer_normalization/Reshape/shapeН
layer_normalization/ReshapeReshapeadd:z:0*layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization/Reshape{
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization/Const
layer_normalization/Fill/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2
layer_normalization/Fill/dimsЖ
layer_normalization/FillFill&layer_normalization/Fill/dims:output:0"layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization/Const_1
layer_normalization/Fill_1/dimsPacklayer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization/Fill_1/dimsО
layer_normalization/Fill_1Fill(layer_normalization/Fill_1/dims:output:0$layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization/Fill_1}
layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_2}
layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization/Const_3Я
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/Fill:output:0#layer_normalization/Fill_1:output:0$layer_normalization/Const_2:output:0$layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2&
$layer_normalization/FusedBatchNormV3Ю
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/Reshape_1У
(layer_normalization/mul_3/ReadVariableOpReadVariableOp1layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization/mul_3/ReadVariableOpЮ
layer_normalization/mul_3Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/mul_3Н
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02(
&layer_normalization/add/ReadVariableOpС
layer_normalization/addAddV2layer_normalization/mul_3:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization/addб
+sequential/dense_4/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_4/Tensordot/ReadVariableOp
!sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_4/Tensordot/axes
!sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_4/Tensordot/free
"sequential/dense_4/Tensordot/ShapeShapelayer_normalization/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/Shape
*sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/GatherV2/axisА
%sequential/dense_4/Tensordot/GatherV2GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/free:output:03sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/GatherV2
,sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_4/Tensordot/GatherV2_1/axisЖ
'sequential/dense_4/Tensordot/GatherV2_1GatherV2+sequential/dense_4/Tensordot/Shape:output:0*sequential/dense_4/Tensordot/axes:output:05sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_4/Tensordot/GatherV2_1
"sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_4/Tensordot/ConstЬ
!sequential/dense_4/Tensordot/ProdProd.sequential/dense_4/Tensordot/GatherV2:output:0+sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_4/Tensordot/Prod
$sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_4/Tensordot/Const_1д
#sequential/dense_4/Tensordot/Prod_1Prod0sequential/dense_4/Tensordot/GatherV2_1:output:0-sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_4/Tensordot/Prod_1
(sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_4/Tensordot/concat/axis
#sequential/dense_4/Tensordot/concatConcatV2*sequential/dense_4/Tensordot/free:output:0*sequential/dense_4/Tensordot/axes:output:01sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_4/Tensordot/concatи
"sequential/dense_4/Tensordot/stackPack*sequential/dense_4/Tensordot/Prod:output:0,sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_4/Tensordot/stackп
&sequential/dense_4/Tensordot/transpose	Transposelayer_normalization/add:z:0,sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2(
&sequential/dense_4/Tensordot/transposeы
$sequential/dense_4/Tensordot/ReshapeReshape*sequential/dense_4/Tensordot/transpose:y:0+sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_4/Tensordot/Reshapeы
#sequential/dense_4/Tensordot/MatMulMatMul-sequential/dense_4/Tensordot/Reshape:output:03sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2%
#sequential/dense_4/Tensordot/MatMul
$sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/dense_4/Tensordot/Const_2
*sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_4/Tensordot/concat_1/axis
%sequential/dense_4/Tensordot/concat_1ConcatV2.sequential/dense_4/Tensordot/GatherV2:output:0-sequential/dense_4/Tensordot/Const_2:output:03sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_4/Tensordot/concat_1н
sequential/dense_4/TensordotReshape-sequential/dense_4/Tensordot/MatMul:product:0.sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/TensordotЦ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)sequential/dense_4/BiasAdd/ReadVariableOpд
sequential/dense_4/BiasAddBiasAdd%sequential/dense_4/Tensordot:output:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22
sequential/dense_4/BiasAddІ
sequential/dense_4/p_re_lu/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22!
sequential/dense_4/p_re_lu/ReluЪ
)sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOp2sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02+
)sequential/dense_4/p_re_lu/ReadVariableOpЄ
sequential/dense_4/p_re_lu/NegNeg1sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22 
sequential/dense_4/p_re_lu/NegЇ
 sequential/dense_4/p_re_lu/Neg_1Neg#sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22"
 sequential/dense_4/p_re_lu/Neg_1Ћ
!sequential/dense_4/p_re_lu/Relu_1Relu$sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22#
!sequential/dense_4/p_re_lu/Relu_1г
sequential/dense_4/p_re_lu/mulMul"sequential/dense_4/p_re_lu/Neg:y:0/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/mulг
sequential/dense_4/p_re_lu/addAddV2-sequential/dense_4/p_re_lu/Relu:activations:0"sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22 
sequential/dense_4/p_re_lu/addб
+sequential/dense_5/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02-
+sequential/dense_5/Tensordot/ReadVariableOp
!sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_5/Tensordot/axes
!sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_5/Tensordot/free
"sequential/dense_5/Tensordot/ShapeShape"sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/Shape
*sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/GatherV2/axisА
%sequential/dense_5/Tensordot/GatherV2GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/free:output:03sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/GatherV2
,sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_5/Tensordot/GatherV2_1/axisЖ
'sequential/dense_5/Tensordot/GatherV2_1GatherV2+sequential/dense_5/Tensordot/Shape:output:0*sequential/dense_5/Tensordot/axes:output:05sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_5/Tensordot/GatherV2_1
"sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_5/Tensordot/ConstЬ
!sequential/dense_5/Tensordot/ProdProd.sequential/dense_5/Tensordot/GatherV2:output:0+sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_5/Tensordot/Prod
$sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_5/Tensordot/Const_1д
#sequential/dense_5/Tensordot/Prod_1Prod0sequential/dense_5/Tensordot/GatherV2_1:output:0-sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_5/Tensordot/Prod_1
(sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_5/Tensordot/concat/axis
#sequential/dense_5/Tensordot/concatConcatV2*sequential/dense_5/Tensordot/free:output:0*sequential/dense_5/Tensordot/axes:output:01sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_5/Tensordot/concatи
"sequential/dense_5/Tensordot/stackPack*sequential/dense_5/Tensordot/Prod:output:0,sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_5/Tensordot/stackц
&sequential/dense_5/Tensordot/transpose	Transpose"sequential/dense_4/p_re_lu/add:z:0,sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22(
&sequential/dense_5/Tensordot/transposeы
$sequential/dense_5/Tensordot/ReshapeReshape*sequential/dense_5/Tensordot/transpose:y:0+sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2&
$sequential/dense_5/Tensordot/Reshapeы
#sequential/dense_5/Tensordot/MatMulMatMul-sequential/dense_5/Tensordot/Reshape:output:03sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2%
#sequential/dense_5/Tensordot/MatMul
$sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2&
$sequential/dense_5/Tensordot/Const_2
*sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_5/Tensordot/concat_1/axis
%sequential/dense_5/Tensordot/concat_1ConcatV2.sequential/dense_5/Tensordot/GatherV2:output:0-sequential/dense_5/Tensordot/Const_2:output:03sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_5/Tensordot/concat_1н
sequential/dense_5/TensordotReshape-sequential/dense_5/Tensordot/MatMul:product:0.sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/TensordotЦ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02+
)sequential/dense_5/BiasAdd/ReadVariableOpд
sequential/dense_5/BiasAddBiasAdd%sequential/dense_5/Tensordot:output:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
sequential/dense_5/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_1/dropout/ConstГ
dropout_1/dropout/MulMul#sequential/dense_5/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/Mul
dropout_1/dropout/ShapeShape#sequential/dense_5/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeз
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_1/dropout/GreaterEqual/yы
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2 
dropout_1/dropout/GreaterEqualЂ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/CastЇ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
dropout_1/dropout/Mul_1
add_1AddV2layer_normalization/add:z:0dropout_1/dropout/Mul_1:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
add_1s
layer_normalization_1/ShapeShape	add_1:z:0*
T0*
_output_shapes
:2
layer_normalization_1/Shape 
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)layer_normalization_1/strided_slice/stackЄ
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_1Є
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice/stack_2ц
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#layer_normalization_1/strided_slice|
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul/xВ
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mulЄ
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_1/stackЈ
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_1Ј
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_1/stack_2№
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_1Б
layer_normalization_1/mul_1Mullayer_normalization_1/mul:z:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_1Є
+layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+layer_normalization_1/strided_slice_2/stackЈ
-layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_1Ј
-layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-layer_normalization_1/strided_slice_2/stack_2№
%layer_normalization_1/strided_slice_2StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_2/stack:output:06layer_normalization_1/strided_slice_2/stack_1:output:06layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%layer_normalization_1/strided_slice_2
layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2
layer_normalization_1/mul_2/xК
layer_normalization_1/mul_2Mul&layer_normalization_1/mul_2/x:output:0.layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2
layer_normalization_1/mul_2
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/0
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2'
%layer_normalization_1/Reshape/shape/3Ђ
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul_1:z:0layer_normalization_1/mul_2:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#layer_normalization_1/Reshape/shapeХ
layer_normalization_1/ReshapeReshape	add_1:z:0,layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2
layer_normalization_1/Reshape
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
layer_normalization_1/Const
layer_normalization_1/Fill/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2!
layer_normalization_1/Fill/dimsО
layer_normalization_1/FillFill(layer_normalization_1/Fill/dims:output:0$layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2
layer_normalization_1/Const_1
!layer_normalization_1/Fill_1/dimsPacklayer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2#
!layer_normalization_1/Fill_1/dimsЦ
layer_normalization_1/Fill_1Fill*layer_normalization_1/Fill_1/dims:output:0&layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2
layer_normalization_1/Fill_1
layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_2
layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2
layer_normalization_1/Const_3н
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/Fill:output:0%layer_normalization_1/Fill_1:output:0&layer_normalization_1/Const_2:output:0&layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2(
&layer_normalization_1/FusedBatchNormV3ж
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2!
layer_normalization_1/Reshape_1Щ
*layer_normalization_1/mul_3/ReadVariableOpReadVariableOp3layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02,
*layer_normalization_1/mul_3/ReadVariableOpж
layer_normalization_1/mul_3Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/mul_3У
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02*
(layer_normalization_1/add/ReadVariableOpЩ
layer_normalization_1/addAddV2layer_normalization_1/mul_3:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
layer_normalization_1/addv
IdentityIdentitylayer_normalization_1/add:z:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*o
_input_shapes^
\:џџџџџџџџџ2Х::::::::::::::::::T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
н
o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66982

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*+
_input_shapes
:џџџџџџџџџ2Х:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs

o
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66993

inputs
identityr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
з
А
#__inference_signature_wrapper_65504
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_640682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1


E__inference_sequential_layer_call_and_return_conditional_losses_64198
dense_4_input
dense_4_64144
dense_4_64146
dense_4_64148
dense_5_64192
dense_5_64194
identityЂdense_4/StatefulPartitionedCallЂdense_5/StatefulPartitionedCallЉ
dense_4/StatefulPartitionedCallStatefulPartitionedCalldense_4_inputdense_4_64144dense_4_64146dense_4_64148*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_641312!
dense_4/StatefulPartitionedCallГ
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_64192dense_5_64194*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ2Х*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_641812!
dense_5/StatefulPartitionedCallХ
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:џџџџџџџџџ2Х:::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:[ W
,
_output_shapes
:џџџџџџџџџ2Х
'
_user_specified_namedense_4_input

­
B__inference_dense_5_layer_call_and_return_conditional_losses_64181

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisб
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axisз
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axisА
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axisН
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2	
BiasAddi
IdentityIdentityBiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :џџџџџџџџџ2:::T P
,
_output_shapes
:џџџџџџџџџ2
 
_user_specified_nameinputs

И
,__inference_functional_1_layer_call_fn_66282

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ	*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_653982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:џџџџџџџџџ2Х
 
_user_specified_nameinputs
Йа

 __inference__wrapped_model_64068
input_1d
`functional_1_transformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resourceb
^functional_1_transformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resourcef
bfunctional_1_transformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resourced
`functional_1_transformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resourcef
bfunctional_1_transformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resourced
`functional_1_transformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resourcef
bfunctional_1_transformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resourced
`functional_1_transformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resourceT
Pfunctional_1_transformer_block_layer_normalization_mul_3_readvariableop_resourceR
Nfunctional_1_transformer_block_layer_normalization_add_readvariableop_resourceW
Sfunctional_1_transformer_block_sequential_dense_4_tensordot_readvariableop_resourceU
Qfunctional_1_transformer_block_sequential_dense_4_biasadd_readvariableop_resourceU
Qfunctional_1_transformer_block_sequential_dense_4_p_re_lu_readvariableop_resourceW
Sfunctional_1_transformer_block_sequential_dense_5_tensordot_readvariableop_resourceU
Qfunctional_1_transformer_block_sequential_dense_5_biasadd_readvariableop_resourceV
Rfunctional_1_transformer_block_layer_normalization_1_mul_3_readvariableop_resourceT
Pfunctional_1_transformer_block_layer_normalization_1_add_readvariableop_resource7
3functional_1_dense_6_matmul_readvariableop_resource8
4functional_1_dense_6_biasadd_readvariableop_resource2
.functional_1_p_re_lu_1_readvariableop_resource7
3functional_1_dense_7_matmul_readvariableop_resource8
4functional_1_dense_7_biasadd_readvariableop_resource
identityЗ
>functional_1/transformer_block/multi_head_self_attention/ShapeShapeinput_1*
T0*
_output_shapes
:2@
>functional_1/transformer_block/multi_head_self_attention/Shapeц
Lfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2N
Lfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stackъ
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2P
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_1ъ
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2P
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_2И
Ffunctional_1/transformer_block/multi_head_self_attention/strided_sliceStridedSliceGfunctional_1/transformer_block/multi_head_self_attention/Shape:output:0Ufunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack:output:0Wfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_1:output:0Wfunctional_1/transformer_block/multi_head_self_attention/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2H
Ffunctional_1/transformer_block/multi_head_self_attention/strided_sliceе
Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpReadVariableOp`functional_1_transformer_block_multi_head_self_attention_dense_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02Y
Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOpш
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2O
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/axesя
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2O
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/freeз
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2P
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Shapeђ
Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis
Qfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2GatherV2Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/free:output:0_functional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2і
Xfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1GatherV2Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Shape:output:0Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0afunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1ъ
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Constќ
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ProdProdZfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2O
Mfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Prodю
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_1
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Prod_1Prod\functional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2_1:output:0Yfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Prod_1ю
Tfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2V
Tfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat/axisы
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concatConcatV2Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/free:output:0Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/axes:output:0]functional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/stackPackVfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Prod:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2P
Nfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/stackЯ
Rfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/transpose	Transposeinput_1Xfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/transpose
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ReshapeReshapeVfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/transpose:y:0Wfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/MatMulMatMulYfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Reshape:output:0_functional_1/transformer_block/multi_head_self_attention/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/MatMulя
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_2ђ
Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axisј
Qfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1ConcatV2Zfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/GatherV2:output:0Yfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/Const_2:output:0_functional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1
Hfunctional_1/transformer_block/multi_head_self_attention/dense/TensordotReshapeYfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/MatMul:product:0Zfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2J
Hfunctional_1/transformer_block/multi_head_self_attention/dense/TensordotЪ
Ufunctional_1/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOpReadVariableOp^functional_1_transformer_block_multi_head_self_attention_dense_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02W
Ufunctional_1/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp
Ffunctional_1/transformer_block/multi_head_self_attention/dense/BiasAddBiasAddQfunctional_1/transformer_block/multi_head_self_attention/dense/Tensordot:output:0]functional_1/transformer_block/multi_head_self_attention/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2H
Ffunctional_1/transformer_block/multi_head_self_attention/dense/BiasAddл
Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpReadVariableOpbfunctional_1_transformer_block_multi_head_self_attention_dense_1_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02[
Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOpь
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/axesѓ
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/freeл
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Shapeі
Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2њ
Zfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis
Ufunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0cfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Ufunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1ю
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ProdProd\functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prodђ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1Prod^functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2_1:output:0[functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1ђ
Vfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axisѕ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concatConcatV2Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/free:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/axes:output:0_functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/stackPackXfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod:output:0Zfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/stackе
Tfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose	Transposeinput_1Zfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2V
Tfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/transposeЃ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeReshapeXfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/transpose:y:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReshapeЃ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulMatMul[functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Reshape:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/MatMulѓ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2і
Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1ConcatV2\functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/GatherV2:output:0[functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/Const_2:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1
Jfunctional_1/transformer_block/multi_head_self_attention/dense_1/TensordotReshape[functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/MatMul:product:0\functional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2L
Jfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordotа
Wfunctional_1/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOpReadVariableOp`functional_1_transformer_block_multi_head_self_attention_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02Y
Wfunctional_1/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp
Hfunctional_1/transformer_block/multi_head_self_attention/dense_1/BiasAddBiasAddSfunctional_1/transformer_block/multi_head_self_attention/dense_1/Tensordot:output:0_functional_1/transformer_block/multi_head_self_attention/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2J
Hfunctional_1/transformer_block/multi_head_self_attention/dense_1/BiasAddл
Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpReadVariableOpbfunctional_1_transformer_block_multi_head_self_attention_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02[
Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOpь
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/axesѓ
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/freeл
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Shapeі
Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2њ
Zfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis
Ufunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0cfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Ufunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1ю
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ProdProd\functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prodђ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1Prod^functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2_1:output:0[functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1ђ
Vfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axisѕ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concatConcatV2Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/free:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/axes:output:0_functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/stackPackXfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod:output:0Zfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/stackе
Tfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose	Transposeinput_1Zfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2V
Tfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/transposeЃ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeReshapeXfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/transpose:y:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReshapeЃ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulMatMul[functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Reshape:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/MatMulѓ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2і
Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1ConcatV2\functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/GatherV2:output:0[functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/Const_2:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1
Jfunctional_1/transformer_block/multi_head_self_attention/dense_2/TensordotReshape[functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/MatMul:product:0\functional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2L
Jfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordotа
Wfunctional_1/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOpReadVariableOp`functional_1_transformer_block_multi_head_self_attention_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02Y
Wfunctional_1/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp
Hfunctional_1/transformer_block/multi_head_self_attention/dense_2/BiasAddBiasAddSfunctional_1/transformer_block/multi_head_self_attention/dense_2/Tensordot:output:0_functional_1/transformer_block/multi_head_self_attention/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2J
Hfunctional_1/transformer_block/multi_head_self_attention/dense_2/BiasAddп
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/1ж
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/2ж
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/3
Ffunctional_1/transformer_block/multi_head_self_attention/Reshape/shapePackOfunctional_1/transformer_block/multi_head_self_attention/strided_slice:output:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/1:output:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/2:output:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2H
Ffunctional_1/transformer_block/multi_head_self_attention/Reshape/shapeє
@functional_1/transformer_block/multi_head_self_attention/ReshapeReshapeOfunctional_1/transformer_block/multi_head_self_attention/dense/BiasAdd:output:0Ofunctional_1/transformer_block/multi_head_self_attention/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2B
@functional_1/transformer_block/multi_head_self_attention/Reshapeы
Gfunctional_1/transformer_block/multi_head_self_attention/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2I
Gfunctional_1/transformer_block/multi_head_self_attention/transpose/permѕ
Bfunctional_1/transformer_block/multi_head_self_attention/transpose	TransposeIfunctional_1/transformer_block/multi_head_self_attention/Reshape:output:0Pfunctional_1/transformer_block/multi_head_self_attention/transpose/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2D
Bfunctional_1/transformer_block/multi_head_self_attention/transposeу
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/1к
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/2к
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/3
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shapePackOfunctional_1/transformer_block/multi_head_self_attention/strided_slice:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/1:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/2:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape/3:output:0*
N*
T0*
_output_shapes
:2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shapeќ
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_1ReshapeQfunctional_1/transformer_block/multi_head_self_attention/dense_1/BiasAdd:output:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape_1/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2D
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_1я
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_1/permConst*
_output_shapes
:*
dtype0*%
valueB"             2K
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_1/perm§
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_1	TransposeKfunctional_1/transformer_block/multi_head_self_attention/Reshape_1:output:0Rfunctional_1/transformer_block/multi_head_self_attention/transpose_1/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2F
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_1у
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/1к
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/2к
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/3
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shapePackOfunctional_1/transformer_block/multi_head_self_attention/strided_slice:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/1:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/2:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape/3:output:0*
N*
T0*
_output_shapes
:2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shapeќ
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_2ReshapeQfunctional_1/transformer_block/multi_head_self_attention/dense_2/BiasAdd:output:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape_2/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2D
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_2я
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_2/permConst*
_output_shapes
:*
dtype0*%
valueB"             2K
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_2/perm§
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_2	TransposeKfunctional_1/transformer_block/multi_head_self_attention/Reshape_2:output:0Rfunctional_1/transformer_block/multi_head_self_attention/transpose_2/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2F
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_2ў
?functional_1/transformer_block/multi_head_self_attention/MatMulBatchMatMulV2Ffunctional_1/transformer_block/multi_head_self_attention/transpose:y:0Hfunctional_1/transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
adj_y(2A
?functional_1/transformer_block/multi_head_self_attention/MatMulќ
@functional_1/transformer_block/multi_head_self_attention/Shape_1ShapeHfunctional_1/transformer_block/multi_head_self_attention/transpose_1:y:0*
T0*
_output_shapes
:2B
@functional_1/transformer_block/multi_head_self_attention/Shape_1ѓ
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ2P
Nfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stackю
Pfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2R
Pfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_1ю
Pfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
Pfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_2Ф
Hfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1StridedSliceIfunctional_1/transformer_block/multi_head_self_attention/Shape_1:output:0Wfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack:output:0Yfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_1:output:0Yfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
Hfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1
=functional_1/transformer_block/multi_head_self_attention/CastCastQfunctional_1/transformer_block/multi_head_self_attention/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2?
=functional_1/transformer_block/multi_head_self_attention/Castъ
=functional_1/transformer_block/multi_head_self_attention/SqrtSqrtAfunctional_1/transformer_block/multi_head_self_attention/Cast:y:0*
T0*
_output_shapes
: 2?
=functional_1/transformer_block/multi_head_self_attention/Sqrtш
@functional_1/transformer_block/multi_head_self_attention/truedivRealDivHfunctional_1/transformer_block/multi_head_self_attention/MatMul:output:0Afunctional_1/transformer_block/multi_head_self_attention/Sqrt:y:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2B
@functional_1/transformer_block/multi_head_self_attention/truedivЁ
@functional_1/transformer_block/multi_head_self_attention/SoftmaxSoftmaxDfunctional_1/transformer_block/multi_head_self_attention/truediv:z:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2B
@functional_1/transformer_block/multi_head_self_attention/Softmax№
Afunctional_1/transformer_block/multi_head_self_attention/MatMul_1BatchMatMulV2Jfunctional_1/transformer_block/multi_head_self_attention/Softmax:softmax:0Hfunctional_1/transformer_block/multi_head_self_attention/transpose_2:y:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2C
Afunctional_1/transformer_block/multi_head_self_attention/MatMul_1я
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_3/permConst*
_output_shapes
:*
dtype0*%
valueB"             2K
Ifunctional_1/transformer_block/multi_head_self_attention/transpose_3/permќ
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_3	TransposeJfunctional_1/transformer_block/multi_head_self_attention/MatMul_1:output:0Rfunctional_1/transformer_block/multi_head_self_attention/transpose_3/perm:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2F
Dfunctional_1/transformer_block/multi_head_self_attention/transpose_3у
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/1л
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Х2L
Jfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/2Х
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shapePackOfunctional_1/transformer_block/multi_head_self_attention/strided_slice:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/1:output:0Sfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape/2:output:0*
N*
T0*
_output_shapes
:2J
Hfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape№
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_3ReshapeHfunctional_1/transformer_block/multi_head_self_attention/transpose_3:y:0Qfunctional_1/transformer_block/multi_head_self_attention/Reshape_3/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2D
Bfunctional_1/transformer_block/multi_head_self_attention/Reshape_3л
Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpReadVariableOpbfunctional_1_transformer_block_multi_head_self_attention_dense_3_tensordot_readvariableop_resource* 
_output_shapes
:
ХХ*
dtype02[
Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOpь
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/axesѓ
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/free
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ShapeShapeKfunctional_1/transformer_block/multi_head_self_attention/Reshape_3:output:0*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Shapeі
Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2њ
Zfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2\
Zfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis
Ufunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1GatherV2Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Shape:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0cfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2W
Ufunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1ю
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ProdProd\functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: 2Q
Ofunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prodђ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1Prod^functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2_1:output:0[functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1ђ
Vfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2X
Vfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axisѕ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concatConcatV2Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/free:output:0Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/axes:output:0_functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/stackPackXfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod:output:0Zfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2R
Pfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/stackЂ
Tfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose	TransposeKfunctional_1/transformer_block/multi_head_self_attention/Reshape_3:output:0Zfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2V
Tfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/transposeЃ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeReshapeXfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/transpose:y:0Yfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReshapeЃ
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulMatMul[functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Reshape:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2S
Qfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/MatMulѓ
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2T
Rfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2і
Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Z
Xfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis
Sfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1ConcatV2\functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/GatherV2:output:0[functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/Const_2:output:0afunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2U
Sfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1
Jfunctional_1/transformer_block/multi_head_self_attention/dense_3/TensordotReshape[functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/MatMul:product:0\functional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot/concat_1:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2L
Jfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordotа
Wfunctional_1/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOpReadVariableOp`functional_1_transformer_block_multi_head_self_attention_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02Y
Wfunctional_1/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp
Hfunctional_1/transformer_block/multi_head_self_attention/dense_3/BiasAddBiasAddSfunctional_1/transformer_block/multi_head_self_attention/dense_3/Tensordot:output:0_functional_1/transformer_block/multi_head_self_attention/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ2J
Hfunctional_1/transformer_block/multi_head_self_attention/dense_3/BiasAdd
/functional_1/transformer_block/dropout/IdentityIdentityQfunctional_1/transformer_block/multi_head_self_attention/dense_3/BiasAdd:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџХ21
/functional_1/transformer_block/dropout/IdentityЫ
"functional_1/transformer_block/addAddV2input_18functional_1/transformer_block/dropout/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2$
"functional_1/transformer_block/addЪ
8functional_1/transformer_block/layer_normalization/ShapeShape&functional_1/transformer_block/add:z:0*
T0*
_output_shapes
:2:
8functional_1/transformer_block/layer_normalization/Shapeк
Ffunctional_1/transformer_block/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Ffunctional_1/transformer_block/layer_normalization/strided_slice/stackо
Hfunctional_1/transformer_block/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hfunctional_1/transformer_block/layer_normalization/strided_slice/stack_1о
Hfunctional_1/transformer_block/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hfunctional_1/transformer_block/layer_normalization/strided_slice/stack_2
@functional_1/transformer_block/layer_normalization/strided_sliceStridedSliceAfunctional_1/transformer_block/layer_normalization/Shape:output:0Ofunctional_1/transformer_block/layer_normalization/strided_slice/stack:output:0Qfunctional_1/transformer_block/layer_normalization/strided_slice/stack_1:output:0Qfunctional_1/transformer_block/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@functional_1/transformer_block/layer_normalization/strided_sliceЖ
8functional_1/transformer_block/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2:
8functional_1/transformer_block/layer_normalization/mul/xІ
6functional_1/transformer_block/layer_normalization/mulMulAfunctional_1/transformer_block/layer_normalization/mul/x:output:0Ifunctional_1/transformer_block/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: 28
6functional_1/transformer_block/layer_normalization/mulо
Hfunctional_1/transformer_block/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hfunctional_1/transformer_block/layer_normalization/strided_slice_1/stackт
Jfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_1т
Jfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_2
Bfunctional_1/transformer_block/layer_normalization/strided_slice_1StridedSliceAfunctional_1/transformer_block/layer_normalization/Shape:output:0Qfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack:output:0Sfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_1:output:0Sfunctional_1/transformer_block/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bfunctional_1/transformer_block/layer_normalization/strided_slice_1Ѕ
8functional_1/transformer_block/layer_normalization/mul_1Mul:functional_1/transformer_block/layer_normalization/mul:z:0Kfunctional_1/transformer_block/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: 2:
8functional_1/transformer_block/layer_normalization/mul_1о
Hfunctional_1/transformer_block/layer_normalization/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hfunctional_1/transformer_block/layer_normalization/strided_slice_2/stackт
Jfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_1т
Jfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_2
Bfunctional_1/transformer_block/layer_normalization/strided_slice_2StridedSliceAfunctional_1/transformer_block/layer_normalization/Shape:output:0Qfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack:output:0Sfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_1:output:0Sfunctional_1/transformer_block/layer_normalization/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bfunctional_1/transformer_block/layer_normalization/strided_slice_2К
:functional_1/transformer_block/layer_normalization/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2<
:functional_1/transformer_block/layer_normalization/mul_2/xЎ
8functional_1/transformer_block/layer_normalization/mul_2MulCfunctional_1/transformer_block/layer_normalization/mul_2/x:output:0Kfunctional_1/transformer_block/layer_normalization/strided_slice_2:output:0*
T0*
_output_shapes
: 2:
8functional_1/transformer_block/layer_normalization/mul_2Ъ
Bfunctional_1/transformer_block/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2D
Bfunctional_1/transformer_block/layer_normalization/Reshape/shape/0Ъ
Bfunctional_1/transformer_block/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2D
Bfunctional_1/transformer_block/layer_normalization/Reshape/shape/3а
@functional_1/transformer_block/layer_normalization/Reshape/shapePackKfunctional_1/transformer_block/layer_normalization/Reshape/shape/0:output:0<functional_1/transformer_block/layer_normalization/mul_1:z:0<functional_1/transformer_block/layer_normalization/mul_2:z:0Kfunctional_1/transformer_block/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2B
@functional_1/transformer_block/layer_normalization/Reshape/shapeЙ
:functional_1/transformer_block/layer_normalization/ReshapeReshape&functional_1/transformer_block/add:z:0Ifunctional_1/transformer_block/layer_normalization/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2<
:functional_1/transformer_block/layer_normalization/ReshapeЙ
8functional_1/transformer_block/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2:
8functional_1/transformer_block/layer_normalization/Const№
<functional_1/transformer_block/layer_normalization/Fill/dimsPack<functional_1/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2>
<functional_1/transformer_block/layer_normalization/Fill/dimsВ
7functional_1/transformer_block/layer_normalization/FillFillEfunctional_1/transformer_block/layer_normalization/Fill/dims:output:0Afunctional_1/transformer_block/layer_normalization/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ29
7functional_1/transformer_block/layer_normalization/FillН
:functional_1/transformer_block/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2<
:functional_1/transformer_block/layer_normalization/Const_1є
>functional_1/transformer_block/layer_normalization/Fill_1/dimsPack<functional_1/transformer_block/layer_normalization/mul_1:z:0*
N*
T0*
_output_shapes
:2@
>functional_1/transformer_block/layer_normalization/Fill_1/dimsК
9functional_1/transformer_block/layer_normalization/Fill_1FillGfunctional_1/transformer_block/layer_normalization/Fill_1/dims:output:0Cfunctional_1/transformer_block/layer_normalization/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2;
9functional_1/transformer_block/layer_normalization/Fill_1Л
:functional_1/transformer_block/layer_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2<
:functional_1/transformer_block/layer_normalization/Const_2Л
:functional_1/transformer_block/layer_normalization/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2<
:functional_1/transformer_block/layer_normalization/Const_3Ј
Cfunctional_1/transformer_block/layer_normalization/FusedBatchNormV3FusedBatchNormV3Cfunctional_1/transformer_block/layer_normalization/Reshape:output:0@functional_1/transformer_block/layer_normalization/Fill:output:0Bfunctional_1/transformer_block/layer_normalization/Fill_1:output:0Cfunctional_1/transformer_block/layer_normalization/Const_2:output:0Cfunctional_1/transformer_block/layer_normalization/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2E
Cfunctional_1/transformer_block/layer_normalization/FusedBatchNormV3Ъ
<functional_1/transformer_block/layer_normalization/Reshape_1ReshapeGfunctional_1/transformer_block/layer_normalization/FusedBatchNormV3:y:0Afunctional_1/transformer_block/layer_normalization/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2>
<functional_1/transformer_block/layer_normalization/Reshape_1 
Gfunctional_1/transformer_block/layer_normalization/mul_3/ReadVariableOpReadVariableOpPfunctional_1_transformer_block_layer_normalization_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02I
Gfunctional_1/transformer_block/layer_normalization/mul_3/ReadVariableOpЪ
8functional_1/transformer_block/layer_normalization/mul_3MulEfunctional_1/transformer_block/layer_normalization/Reshape_1:output:0Ofunctional_1/transformer_block/layer_normalization/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2:
8functional_1/transformer_block/layer_normalization/mul_3
Efunctional_1/transformer_block/layer_normalization/add/ReadVariableOpReadVariableOpNfunctional_1_transformer_block_layer_normalization_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02G
Efunctional_1/transformer_block/layer_normalization/add/ReadVariableOpН
6functional_1/transformer_block/layer_normalization/addAddV2<functional_1/transformer_block/layer_normalization/mul_3:z:0Mfunctional_1/transformer_block/layer_normalization/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х28
6functional_1/transformer_block/layer_normalization/addЎ
Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/ReadVariableOpReadVariableOpSfunctional_1_transformer_block_sequential_dense_4_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02L
Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/ReadVariableOpЮ
@functional_1/transformer_block/sequential/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@functional_1/transformer_block/sequential/dense_4/Tensordot/axesе
@functional_1/transformer_block/sequential/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@functional_1/transformer_block/sequential/dense_4/Tensordot/free№
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/ShapeShape:functional_1/transformer_block/layer_normalization/add:z:0*
T0*
_output_shapes
:2C
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/Shapeи
Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2/axisЫ
Dfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2GatherV2Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/Shape:output:0Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/free:output:0Rfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2м
Kfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axisб
Ffunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1GatherV2Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/Shape:output:0Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/axes:output:0Tfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ffunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1а
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/ConstШ
@functional_1/transformer_block/sequential/dense_4/Tensordot/ProdProdMfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@functional_1/transformer_block/sequential/dense_4/Tensordot/Prodд
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_1а
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/Prod_1ProdOfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2_1:output:0Lfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/Prod_1д
Gfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat/axisЊ
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/concatConcatV2Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/free:output:0Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/axes:output:0Pfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/concatд
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/stackPackIfunctional_1/transformer_block/sequential/dense_4/Tensordot/Prod:output:0Kfunctional_1/transformer_block/sequential/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Afunctional_1/transformer_block/sequential/dense_4/Tensordot/stackл
Efunctional_1/transformer_block/sequential/dense_4/Tensordot/transpose	Transpose:functional_1/transformer_block/layer_normalization/add:z:0Kfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2G
Efunctional_1/transformer_block/sequential/dense_4/Tensordot/transposeч
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/ReshapeReshapeIfunctional_1/transformer_block/sequential/dense_4/Tensordot/transpose:y:0Jfunctional_1/transformer_block/sequential/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2E
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/Reshapeч
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/MatMulMatMulLfunctional_1/transformer_block/sequential/dense_4/Tensordot/Reshape:output:0Rfunctional_1/transformer_block/sequential/dense_4/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2D
Bfunctional_1/transformer_block/sequential/dense_4/Tensordot/MatMulе
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_2и
Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Ifunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1/axisЗ
Dfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1ConcatV2Mfunctional_1/transformer_block/sequential/dense_4/Tensordot/GatherV2:output:0Lfunctional_1/transformer_block/sequential/dense_4/Tensordot/Const_2:output:0Rfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1й
;functional_1/transformer_block/sequential/dense_4/TensordotReshapeLfunctional_1/transformer_block/sequential/dense_4/Tensordot/MatMul:product:0Mfunctional_1/transformer_block/sequential/dense_4/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22=
;functional_1/transformer_block/sequential/dense_4/TensordotЃ
Hfunctional_1/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOpQfunctional_1_transformer_block_sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02J
Hfunctional_1/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOpа
9functional_1/transformer_block/sequential/dense_4/BiasAddBiasAddDfunctional_1/transformer_block/sequential/dense_4/Tensordot:output:0Pfunctional_1/transformer_block/sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ22;
9functional_1/transformer_block/sequential/dense_4/BiasAdd
>functional_1/transformer_block/sequential/dense_4/p_re_lu/ReluReluBfunctional_1/transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22@
>functional_1/transformer_block/sequential/dense_4/p_re_lu/ReluЇ
Hfunctional_1/transformer_block/sequential/dense_4/p_re_lu/ReadVariableOpReadVariableOpQfunctional_1_transformer_block_sequential_dense_4_p_re_lu_readvariableop_resource*
_output_shapes
:	2*
dtype02J
Hfunctional_1/transformer_block/sequential/dense_4/p_re_lu/ReadVariableOp
=functional_1/transformer_block/sequential/dense_4/p_re_lu/NegNegPfunctional_1/transformer_block/sequential/dense_4/p_re_lu/ReadVariableOp:value:0*
T0*
_output_shapes
:	22?
=functional_1/transformer_block/sequential/dense_4/p_re_lu/Neg
?functional_1/transformer_block/sequential/dense_4/p_re_lu/Neg_1NegBfunctional_1/transformer_block/sequential/dense_4/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22A
?functional_1/transformer_block/sequential/dense_4/p_re_lu/Neg_1
@functional_1/transformer_block/sequential/dense_4/p_re_lu/Relu_1ReluCfunctional_1/transformer_block/sequential/dense_4/p_re_lu/Neg_1:y:0*
T0*,
_output_shapes
:џџџџџџџџџ22B
@functional_1/transformer_block/sequential/dense_4/p_re_lu/Relu_1Я
=functional_1/transformer_block/sequential/dense_4/p_re_lu/mulMulAfunctional_1/transformer_block/sequential/dense_4/p_re_lu/Neg:y:0Nfunctional_1/transformer_block/sequential/dense_4/p_re_lu/Relu_1:activations:0*
T0*,
_output_shapes
:џџџџџџџџџ22?
=functional_1/transformer_block/sequential/dense_4/p_re_lu/mulЯ
=functional_1/transformer_block/sequential/dense_4/p_re_lu/addAddV2Lfunctional_1/transformer_block/sequential/dense_4/p_re_lu/Relu:activations:0Afunctional_1/transformer_block/sequential/dense_4/p_re_lu/mul:z:0*
T0*,
_output_shapes
:џџџџџџџџџ22?
=functional_1/transformer_block/sequential/dense_4/p_re_lu/addЎ
Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/ReadVariableOpReadVariableOpSfunctional_1_transformer_block_sequential_dense_5_tensordot_readvariableop_resource* 
_output_shapes
:
Х*
dtype02L
Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/ReadVariableOpЮ
@functional_1/transformer_block/sequential/dense_5/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2B
@functional_1/transformer_block/sequential/dense_5/Tensordot/axesе
@functional_1/transformer_block/sequential/dense_5/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2B
@functional_1/transformer_block/sequential/dense_5/Tensordot/freeї
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/ShapeShapeAfunctional_1/transformer_block/sequential/dense_4/p_re_lu/add:z:0*
T0*
_output_shapes
:2C
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/Shapeи
Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2/axisЫ
Dfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2GatherV2Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/Shape:output:0Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/free:output:0Rfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2F
Dfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2м
Kfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2M
Kfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axisб
Ffunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1GatherV2Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/Shape:output:0Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/axes:output:0Tfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2H
Ffunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1а
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2C
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/ConstШ
@functional_1/transformer_block/sequential/dense_5/Tensordot/ProdProdMfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const:output:0*
T0*
_output_shapes
: 2B
@functional_1/transformer_block/sequential/dense_5/Tensordot/Prodд
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2E
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_1а
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/Prod_1ProdOfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2_1:output:0Lfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2D
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/Prod_1д
Gfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2I
Gfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat/axisЊ
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/concatConcatV2Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/free:output:0Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/axes:output:0Pfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2D
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/concatд
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/stackPackIfunctional_1/transformer_block/sequential/dense_5/Tensordot/Prod:output:0Kfunctional_1/transformer_block/sequential/dense_5/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2C
Afunctional_1/transformer_block/sequential/dense_5/Tensordot/stackт
Efunctional_1/transformer_block/sequential/dense_5/Tensordot/transpose	TransposeAfunctional_1/transformer_block/sequential/dense_4/p_re_lu/add:z:0Kfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ22G
Efunctional_1/transformer_block/sequential/dense_5/Tensordot/transposeч
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/ReshapeReshapeIfunctional_1/transformer_block/sequential/dense_5/Tensordot/transpose:y:0Jfunctional_1/transformer_block/sequential/dense_5/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2E
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/Reshapeч
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/MatMulMatMulLfunctional_1/transformer_block/sequential/dense_5/Tensordot/Reshape:output:0Rfunctional_1/transformer_block/sequential/dense_5/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџХ2D
Bfunctional_1/transformer_block/sequential/dense_5/Tensordot/MatMulе
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Х2E
Cfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_2и
Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2K
Ifunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1/axisЗ
Dfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1ConcatV2Mfunctional_1/transformer_block/sequential/dense_5/Tensordot/GatherV2:output:0Lfunctional_1/transformer_block/sequential/dense_5/Tensordot/Const_2:output:0Rfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2F
Dfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1й
;functional_1/transformer_block/sequential/dense_5/TensordotReshapeLfunctional_1/transformer_block/sequential/dense_5/Tensordot/MatMul:product:0Mfunctional_1/transformer_block/sequential/dense_5/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2=
;functional_1/transformer_block/sequential/dense_5/TensordotЃ
Hfunctional_1/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOpQfunctional_1_transformer_block_sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:Х*
dtype02J
Hfunctional_1/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOpа
9functional_1/transformer_block/sequential/dense_5/BiasAddBiasAddDfunctional_1/transformer_block/sequential/dense_5/Tensordot:output:0Pfunctional_1/transformer_block/sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2;
9functional_1/transformer_block/sequential/dense_5/BiasAddэ
1functional_1/transformer_block/dropout_1/IdentityIdentityBfunctional_1/transformer_block/sequential/dense_5/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х23
1functional_1/transformer_block/dropout_1/Identity
$functional_1/transformer_block/add_1AddV2:functional_1/transformer_block/layer_normalization/add:z:0:functional_1/transformer_block/dropout_1/Identity:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2&
$functional_1/transformer_block/add_1а
:functional_1/transformer_block/layer_normalization_1/ShapeShape(functional_1/transformer_block/add_1:z:0*
T0*
_output_shapes
:2<
:functional_1/transformer_block/layer_normalization_1/Shapeо
Hfunctional_1/transformer_block/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hfunctional_1/transformer_block/layer_normalization_1/strided_slice/stackт
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_1т
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_2 
Bfunctional_1/transformer_block/layer_normalization_1/strided_sliceStridedSliceCfunctional_1/transformer_block/layer_normalization_1/Shape:output:0Qfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack:output:0Sfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_1:output:0Sfunctional_1/transformer_block/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2D
Bfunctional_1/transformer_block/layer_normalization_1/strided_sliceК
:functional_1/transformer_block/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :2<
:functional_1/transformer_block/layer_normalization_1/mul/xЎ
8functional_1/transformer_block/layer_normalization_1/mulMulCfunctional_1/transformer_block/layer_normalization_1/mul/x:output:0Kfunctional_1/transformer_block/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: 2:
8functional_1/transformer_block/layer_normalization_1/mulт
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stackц
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_1ц
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_2Њ
Dfunctional_1/transformer_block/layer_normalization_1/strided_slice_1StridedSliceCfunctional_1/transformer_block/layer_normalization_1/Shape:output:0Sfunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack:output:0Ufunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_1:output:0Ufunctional_1/transformer_block/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dfunctional_1/transformer_block/layer_normalization_1/strided_slice_1­
:functional_1/transformer_block/layer_normalization_1/mul_1Mul<functional_1/transformer_block/layer_normalization_1/mul:z:0Mfunctional_1/transformer_block/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: 2<
:functional_1/transformer_block/layer_normalization_1/mul_1т
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stackц
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_1ц
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_2Њ
Dfunctional_1/transformer_block/layer_normalization_1/strided_slice_2StridedSliceCfunctional_1/transformer_block/layer_normalization_1/Shape:output:0Sfunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack:output:0Ufunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_1:output:0Ufunctional_1/transformer_block/layer_normalization_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dfunctional_1/transformer_block/layer_normalization_1/strided_slice_2О
<functional_1/transformer_block/layer_normalization_1/mul_2/xConst*
_output_shapes
: *
dtype0*
value	B :2>
<functional_1/transformer_block/layer_normalization_1/mul_2/xЖ
:functional_1/transformer_block/layer_normalization_1/mul_2MulEfunctional_1/transformer_block/layer_normalization_1/mul_2/x:output:0Mfunctional_1/transformer_block/layer_normalization_1/strided_slice_2:output:0*
T0*
_output_shapes
: 2<
:functional_1/transformer_block/layer_normalization_1/mul_2Ю
Dfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2F
Dfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/0Ю
Dfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2F
Dfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/3м
Bfunctional_1/transformer_block/layer_normalization_1/Reshape/shapePackMfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/0:output:0>functional_1/transformer_block/layer_normalization_1/mul_1:z:0>functional_1/transformer_block/layer_normalization_1/mul_2:z:0Mfunctional_1/transformer_block/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2D
Bfunctional_1/transformer_block/layer_normalization_1/Reshape/shapeС
<functional_1/transformer_block/layer_normalization_1/ReshapeReshape(functional_1/transformer_block/add_1:z:0Kfunctional_1/transformer_block/layer_normalization_1/Reshape/shape:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ2>
<functional_1/transformer_block/layer_normalization_1/ReshapeН
:functional_1/transformer_block/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:functional_1/transformer_block/layer_normalization_1/Constі
>functional_1/transformer_block/layer_normalization_1/Fill/dimsPack>functional_1/transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2@
>functional_1/transformer_block/layer_normalization_1/Fill/dimsК
9functional_1/transformer_block/layer_normalization_1/FillFillGfunctional_1/transformer_block/layer_normalization_1/Fill/dims:output:0Cfunctional_1/transformer_block/layer_normalization_1/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2;
9functional_1/transformer_block/layer_normalization_1/FillС
<functional_1/transformer_block/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    2>
<functional_1/transformer_block/layer_normalization_1/Const_1њ
@functional_1/transformer_block/layer_normalization_1/Fill_1/dimsPack>functional_1/transformer_block/layer_normalization_1/mul_1:z:0*
N*
T0*
_output_shapes
:2B
@functional_1/transformer_block/layer_normalization_1/Fill_1/dimsТ
;functional_1/transformer_block/layer_normalization_1/Fill_1FillIfunctional_1/transformer_block/layer_normalization_1/Fill_1/dims:output:0Efunctional_1/transformer_block/layer_normalization_1/Const_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ2=
;functional_1/transformer_block/layer_normalization_1/Fill_1П
<functional_1/transformer_block/layer_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2>
<functional_1/transformer_block/layer_normalization_1/Const_2П
<functional_1/transformer_block/layer_normalization_1/Const_3Const*
_output_shapes
: *
dtype0*
valueB 2>
<functional_1/transformer_block/layer_normalization_1/Const_3Ж
Efunctional_1/transformer_block/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3Efunctional_1/transformer_block/layer_normalization_1/Reshape:output:0Bfunctional_1/transformer_block/layer_normalization_1/Fill:output:0Dfunctional_1/transformer_block/layer_normalization_1/Fill_1:output:0Efunctional_1/transformer_block/layer_normalization_1/Const_2:output:0Efunctional_1/transformer_block/layer_normalization_1/Const_3:output:0*
T0*
U0*x
_output_shapesf
d:"џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:*
data_formatNCHW*
epsilon%o:2G
Efunctional_1/transformer_block/layer_normalization_1/FusedBatchNormV3в
>functional_1/transformer_block/layer_normalization_1/Reshape_1ReshapeIfunctional_1/transformer_block/layer_normalization_1/FusedBatchNormV3:y:0Cfunctional_1/transformer_block/layer_normalization_1/Shape:output:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2@
>functional_1/transformer_block/layer_normalization_1/Reshape_1І
Ifunctional_1/transformer_block/layer_normalization_1/mul_3/ReadVariableOpReadVariableOpRfunctional_1_transformer_block_layer_normalization_1_mul_3_readvariableop_resource*
_output_shapes	
:Х*
dtype02K
Ifunctional_1/transformer_block/layer_normalization_1/mul_3/ReadVariableOpв
:functional_1/transformer_block/layer_normalization_1/mul_3MulGfunctional_1/transformer_block/layer_normalization_1/Reshape_1:output:0Qfunctional_1/transformer_block/layer_normalization_1/mul_3/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2<
:functional_1/transformer_block/layer_normalization_1/mul_3 
Gfunctional_1/transformer_block/layer_normalization_1/add/ReadVariableOpReadVariableOpPfunctional_1_transformer_block_layer_normalization_1_add_readvariableop_resource*
_output_shapes	
:Х*
dtype02I
Gfunctional_1/transformer_block/layer_normalization_1/add/ReadVariableOpХ
8functional_1/transformer_block/layer_normalization_1/addAddV2>functional_1/transformer_block/layer_normalization_1/mul_3:z:0Ofunctional_1/transformer_block/layer_normalization_1/add/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџ2Х2:
8functional_1/transformer_block/layer_normalization_1/addО
<functional_1/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2>
<functional_1/global_average_pooling1d/Mean/reduction_indices
*functional_1/global_average_pooling1d/MeanMean<functional_1/transformer_block/layer_normalization_1/add:z:0Efunctional_1/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2,
*functional_1/global_average_pooling1d/MeanЖ
functional_1/dropout_2/IdentityIdentity3functional_1/global_average_pooling1d/Mean:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2!
functional_1/dropout_2/IdentityЮ
*functional_1/dense_6/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
Х*
dtype02,
*functional_1/dense_6/MatMul/ReadVariableOpе
functional_1/dense_6/MatMulMatMul(functional_1/dropout_2/Identity:output:02functional_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense_6/MatMulЬ
+functional_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_1/dense_6/BiasAdd/ReadVariableOpж
functional_1/dense_6/BiasAddBiasAdd%functional_1/dense_6/MatMul:product:03functional_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/dense_6/BiasAdd
functional_1/p_re_lu_1/ReluRelu%functional_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/p_re_lu_1/ReluК
%functional_1/p_re_lu_1/ReadVariableOpReadVariableOp.functional_1_p_re_lu_1_readvariableop_resource*
_output_shapes	
:*
dtype02'
%functional_1/p_re_lu_1/ReadVariableOp
functional_1/p_re_lu_1/NegNeg-functional_1/p_re_lu_1/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
functional_1/p_re_lu_1/Neg
functional_1/p_re_lu_1/Neg_1Neg%functional_1/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/p_re_lu_1/Neg_1
functional_1/p_re_lu_1/Relu_1Relu functional_1/p_re_lu_1/Neg_1:y:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/p_re_lu_1/Relu_1П
functional_1/p_re_lu_1/mulMulfunctional_1/p_re_lu_1/Neg:y:0+functional_1/p_re_lu_1/Relu_1:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/p_re_lu_1/mulП
functional_1/p_re_lu_1/addAddV2)functional_1/p_re_lu_1/Relu:activations:0functional_1/p_re_lu_1/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_1/p_re_lu_1/addЁ
functional_1/dropout_3/IdentityIdentityfunctional_1/p_re_lu_1/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
functional_1/dropout_3/IdentityЭ
*functional_1/dense_7/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_7_matmul_readvariableop_resource*
_output_shapes
:		*
dtype02,
*functional_1/dense_7/MatMul/ReadVariableOpд
functional_1/dense_7/MatMulMatMul(functional_1/dropout_3/Identity:output:02functional_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
functional_1/dense_7/MatMulЫ
+functional_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02-
+functional_1/dense_7/BiasAdd/ReadVariableOpе
functional_1/dense_7/BiasAddBiasAdd%functional_1/dense_7/MatMul:product:03functional_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
functional_1/dense_7/BiasAdd 
functional_1/dense_7/SoftmaxSoftmax%functional_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ	2
functional_1/dense_7/Softmaxz
IdentityIdentity&functional_1/dense_7/Softmax:softmax:0*
T0*'
_output_shapes
:џџџџџџџџџ	2

Identity"
identityIdentity:output:0*
_input_shapesr
p:џџџџџџџџџ2Х:::::::::::::::::::::::U Q
,
_output_shapes
:џџџџџџџџџ2Х
!
_user_specified_name	input_1
Ы
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_65081

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџХ:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs

E
)__inference_dropout_2_layer_call_fn_67025

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџХ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_650812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџХ2

Identity"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџХ:P L
(
_output_shapes
:џџџџџџџџџХ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Џ
serving_default
@
input_15
serving_default_input_1:0џџџџџџџџџ2Х;
dense_70
StatefulPartitionedCall:0џџџџџџџџџ	tensorflow/serving/predict:њи
в
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
Ћ_default_save_signature
+Ќ&call_and_return_all_conditional_losses
­__call__"Д
_tf_keras_network{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 325]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TransformerBlock", "config": {"layer was saved without config": true}, "name": "transformer_block", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling1D", "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling1d", "inbound_nodes": [[["transformer_block", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_2", "inbound_nodes": [[["global_average_pooling1d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dropout_2", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_1", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout_3", "inbound_nodes": [[["p_re_lu_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dropout_3", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_7", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional"}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѕ"ђ
_tf_keras_input_layerв{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 325]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 325]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

att
ffn

layernorm1

layernorm2
dropout1
dropout2
	variables
regularization_losses
trainable_variables
	keras_api
+Ў&call_and_return_all_conditional_losses
Џ__call__"Ѓ
_tf_keras_layer{"class_name": "TransformerBlock", "name": "transformer_block", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}

	variables
regularization_losses
trainable_variables
	keras_api
+А&call_and_return_all_conditional_losses
Б__call__"
_tf_keras_layerъ{"class_name": "GlobalAveragePooling1D", "name": "global_average_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ч
	variables
regularization_losses
trainable_variables
 	keras_api
+В&call_and_return_all_conditional_losses
Г__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ї

!kernel
"bias
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+Д&call_and_return_all_conditional_losses
Е__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 325]}}
Ё
	'alpha
(	variables
)regularization_losses
*trainable_variables
+	keras_api
+Ж&call_and_return_all_conditional_losses
З__call__"
_tf_keras_layerы{"class_name": "PReLU", "name": "p_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ч
,	variables
-regularization_losses
.trainable_variables
/	keras_api
+И&call_and_return_all_conditional_losses
Й__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
і

0kernel
1bias
2	variables
3regularization_losses
4trainable_variables
5	keras_api
+К&call_and_return_all_conditional_losses
Л__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 9, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}

6iter

7beta_1

8beta_2
	9decay
:learning_rate!mџ"m'm0m1m;m<m=m>m?m@mAmBmCmDmEmFmGmHmImJmKm!v"v'v0v1v;v<v=v>v?v@vAv BvЁCvЂDvЃEvЄFvЅGvІHvЇIvЈJvЉKvЊ"
	optimizer
Ц
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
!17
"18
'19
020
121"
trackable_list_wrapper
 "
trackable_list_wrapper
Ц
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
!17
"18
'19
020
121"
trackable_list_wrapper
Ю
Llayer_regularization_losses
Mlayer_metrics

Nlayers

	variables
regularization_losses
Ometrics
Pnon_trainable_variables
trainable_variables
­__call__
Ћ_default_save_signature
+Ќ&call_and_return_all_conditional_losses
'Ќ"call_and_return_conditional_losses"
_generic_user_object
-
Мserving_default"
signature_map

Qquery_dense
R	key_dense
Svalue_dense
Tcombine_heads
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"В
_tf_keras_layer{"class_name": "MultiHeadSelfAttention", "name": "multi_head_self_attention", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
н
Ylayer_with_weights-0
Ylayer-0
Zlayer_with_weights-1
Zlayer-1
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"ў
_tf_keras_sequentialп{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 325]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 325]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_4_input"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
р
_axis
	Hgamma
Ibeta
`	variables
aregularization_losses
btrainable_variables
c	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"А
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}
ф
daxis
	Jgamma
Kbeta
e	variables
fregularization_losses
gtrainable_variables
h	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"Д
_tf_keras_layer{"class_name": "LayerNormalization", "name": "layer_normalization_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "layer_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}
у
i	variables
jregularization_losses
ktrainable_variables
l	keras_api
+Х&call_and_return_all_conditional_losses
Ц__call__"в
_tf_keras_layerИ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ч
m	variables
nregularization_losses
otrainable_variables
p	keras_api
+Ч&call_and_return_all_conditional_losses
Ш__call__"ж
_tf_keras_layerМ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}

;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16"
trackable_list_wrapper
 "
trackable_list_wrapper

;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16"
trackable_list_wrapper
А
qlayer_regularization_losses
rlayer_metrics

slayers
	variables
regularization_losses
tmetrics
unon_trainable_variables
trainable_variables
Џ__call__
+Ў&call_and_return_all_conditional_losses
'Ў"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
vlayer_regularization_losses
wlayer_metrics

xlayers
	variables
regularization_losses
ymetrics
znon_trainable_variables
trainable_variables
Б__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
{layer_regularization_losses
|layer_metrics

}layers
	variables
regularization_losses
~metrics
non_trainable_variables
trainable_variables
Г__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
": 
Х2dense_6/kernel
:2dense_6/bias
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
Е
 layer_regularization_losses
layer_metrics
layers
#	variables
$regularization_losses
metrics
non_trainable_variables
%trainable_variables
Е__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
:2p_re_lu_1/alpha
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
Е
 layer_regularization_losses
layer_metrics
layers
(	variables
)regularization_losses
metrics
non_trainable_variables
*trainable_variables
З__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 layer_regularization_losses
layer_metrics
layers
,	variables
-regularization_losses
metrics
non_trainable_variables
.trainable_variables
Й__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
!:		2dense_7/kernel
:	2dense_7/bias
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
Е
 layer_regularization_losses
layer_metrics
layers
2	variables
3regularization_losses
metrics
non_trainable_variables
4trainable_variables
Л__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
L:J
ХХ28transformer_block/multi_head_self_attention/dense/kernel
E:CХ26transformer_block/multi_head_self_attention/dense/bias
N:L
ХХ2:transformer_block/multi_head_self_attention/dense_1/kernel
G:EХ28transformer_block/multi_head_self_attention/dense_1/bias
N:L
ХХ2:transformer_block/multi_head_self_attention/dense_2/kernel
G:EХ28transformer_block/multi_head_self_attention/dense_2/bias
N:L
ХХ2:transformer_block/multi_head_self_attention/dense_3/kernel
G:EХ28transformer_block/multi_head_self_attention/dense_3/bias
?:=
Х2+transformer_block/sequential/dense_4/kernel
8:62)transformer_block/sequential/dense_4/bias
E:C	222transformer_block/sequential/dense_4/p_re_lu/alpha
?:=
Х2+transformer_block/sequential/dense_5/kernel
8:6Х2)transformer_block/sequential/dense_5/bias
::8Х2+transformer_block/layer_normalization/gamma
9:7Х2*transformer_block/layer_normalization/beta
<::Х2-transformer_block/layer_normalization_1/gamma
;:9Х2,transformer_block/layer_normalization_1/beta
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ћ

;kernel
<bias
	variables
regularization_losses
trainable_variables
	keras_api
+Щ&call_and_return_all_conditional_losses
Ъ__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}
џ

=kernel
>bias
	variables
regularization_losses
trainable_variables
	keras_api
+Ы&call_and_return_all_conditional_losses
Ь__call__"д
_tf_keras_layerК{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}
џ

?kernel
@bias
	variables
regularization_losses
 trainable_variables
Ё	keras_api
+Э&call_and_return_all_conditional_losses
Ю__call__"д
_tf_keras_layerК{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}


Akernel
Bbias
Ђ	variables
Ѓregularization_losses
Єtrainable_variables
Ѕ	keras_api
+Я&call_and_return_all_conditional_losses
а__call__"ж
_tf_keras_layerМ{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 325]}}
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
;0
<1
=2
>3
?4
@5
A6
B7"
trackable_list_wrapper
Е
 Іlayer_regularization_losses
Їlayer_metrics
Јlayers
U	variables
Vregularization_losses
Љmetrics
Њnon_trainable_variables
Wtrainable_variables
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object


Ћ
activation
Ќ_inbound_nodes

Ckernel
Dbias
­	variables
Ўregularization_losses
Џtrainable_variables
А	keras_api
+б&call_and_return_all_conditional_losses
в__call__"Џ
_tf_keras_layer{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 325}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 325]}}

Б_inbound_nodes

Fkernel
Gbias
В	variables
Гregularization_losses
Дtrainable_variables
Е	keras_api
+г&call_and_return_all_conditional_losses
д__call__"д
_tf_keras_layerК{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 325, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 512]}}
C
C0
D1
E2
F3
G4"
trackable_list_wrapper
 "
trackable_list_wrapper
C
C0
D1
E2
F3
G4"
trackable_list_wrapper
Е
 Жlayer_regularization_losses
Зlayer_metrics
Иlayers
[	variables
\regularization_losses
Йmetrics
Кnon_trainable_variables
]trainable_variables
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
Е
 Лlayer_regularization_losses
Мlayer_metrics
Нlayers
`	variables
aregularization_losses
Оmetrics
Пnon_trainable_variables
btrainable_variables
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
Е
 Рlayer_regularization_losses
Сlayer_metrics
Тlayers
e	variables
fregularization_losses
Уmetrics
Фnon_trainable_variables
gtrainable_variables
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Хlayer_regularization_losses
Цlayer_metrics
Чlayers
i	variables
jregularization_losses
Шmetrics
Щnon_trainable_variables
ktrainable_variables
Ц__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
 Ъlayer_regularization_losses
Ыlayer_metrics
Ьlayers
m	variables
nregularization_losses
Эmetrics
Юnon_trainable_variables
otrainable_variables
Ш__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
П

Яtotal

аcount
б	variables
в	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


гtotal

дcount
е
_fn_kwargs
ж	variables
з	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
И
 иlayer_regularization_losses
йlayer_metrics
кlayers
	variables
regularization_losses
лmetrics
мnon_trainable_variables
trainable_variables
Ъ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
И
 нlayer_regularization_losses
оlayer_metrics
пlayers
	variables
regularization_losses
рmetrics
сnon_trainable_variables
trainable_variables
Ь__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
И
 тlayer_regularization_losses
уlayer_metrics
фlayers
	variables
regularization_losses
хmetrics
цnon_trainable_variables
 trainable_variables
Ю__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
И
 чlayer_regularization_losses
шlayer_metrics
щlayers
Ђ	variables
Ѓregularization_losses
ъmetrics
ыnon_trainable_variables
Єtrainable_variables
а__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ѕ
	Ealpha
ь	variables
эregularization_losses
юtrainable_variables
я	keras_api
+е&call_and_return_all_conditional_losses
ж__call__"
_tf_keras_layerы{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 512]}}
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
C0
D1
E2"
trackable_list_wrapper
И
 №layer_regularization_losses
ёlayer_metrics
ђlayers
­	variables
Ўregularization_losses
ѓmetrics
єnon_trainable_variables
Џtrainable_variables
в__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
И
 ѕlayer_regularization_losses
іlayer_metrics
їlayers
В	variables
Гregularization_losses
јmetrics
љnon_trainable_variables
Дtrainable_variables
д__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Я0
а1"
trackable_list_wrapper
.
б	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
г0
д1"
trackable_list_wrapper
.
ж	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
E0"
trackable_list_wrapper
И
 њlayer_regularization_losses
ћlayer_metrics
ќlayers
ь	variables
эregularization_losses
§metrics
ўnon_trainable_variables
юtrainable_variables
ж__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
Ћ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
':%
Х2Adam/dense_6/kernel/m
 :2Adam/dense_6/bias/m
#:!2Adam/p_re_lu_1/alpha/m
&:$		2Adam/dense_7/kernel/m
:	2Adam/dense_7/bias/m
Q:O
ХХ2?Adam/transformer_block/multi_head_self_attention/dense/kernel/m
J:HХ2=Adam/transformer_block/multi_head_self_attention/dense/bias/m
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/m
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_1/bias/m
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/m
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_2/bias/m
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/m
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_3/bias/m
D:B
Х22Adam/transformer_block/sequential/dense_4/kernel/m
=:;20Adam/transformer_block/sequential/dense_4/bias/m
J:H	229Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/m
D:B
Х22Adam/transformer_block/sequential/dense_5/kernel/m
=:;Х20Adam/transformer_block/sequential/dense_5/bias/m
?:=Х22Adam/transformer_block/layer_normalization/gamma/m
>:<Х21Adam/transformer_block/layer_normalization/beta/m
A:?Х24Adam/transformer_block/layer_normalization_1/gamma/m
@:>Х23Adam/transformer_block/layer_normalization_1/beta/m
':%
Х2Adam/dense_6/kernel/v
 :2Adam/dense_6/bias/v
#:!2Adam/p_re_lu_1/alpha/v
&:$		2Adam/dense_7/kernel/v
:	2Adam/dense_7/bias/v
Q:O
ХХ2?Adam/transformer_block/multi_head_self_attention/dense/kernel/v
J:HХ2=Adam/transformer_block/multi_head_self_attention/dense/bias/v
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_1/kernel/v
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_1/bias/v
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_2/kernel/v
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_2/bias/v
S:Q
ХХ2AAdam/transformer_block/multi_head_self_attention/dense_3/kernel/v
L:JХ2?Adam/transformer_block/multi_head_self_attention/dense_3/bias/v
D:B
Х22Adam/transformer_block/sequential/dense_4/kernel/v
=:;20Adam/transformer_block/sequential/dense_4/bias/v
J:H	229Adam/transformer_block/sequential/dense_4/p_re_lu/alpha/v
D:B
Х22Adam/transformer_block/sequential/dense_5/kernel/v
=:;Х20Adam/transformer_block/sequential/dense_5/bias/v
?:=Х22Adam/transformer_block/layer_normalization/gamma/v
>:<Х21Adam/transformer_block/layer_normalization/beta/v
A:?Х24Adam/transformer_block/layer_normalization_1/gamma/v
@:>Х23Adam/transformer_block/layer_normalization_1/beta/v
у2р
 __inference__wrapped_model_64068Л
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *+Ђ(
&#
input_1џџџџџџџџџ2Х
ъ2ч
G__inference_functional_1_layer_call_and_return_conditional_losses_65858
G__inference_functional_1_layer_call_and_return_conditional_losses_66184
G__inference_functional_1_layer_call_and_return_conditional_losses_65181
G__inference_functional_1_layer_call_and_return_conditional_losses_65236Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ў2ћ
,__inference_functional_1_layer_call_fn_65341
,__inference_functional_1_layer_call_fn_66282
,__inference_functional_1_layer_call_fn_66233
,__inference_functional_1_layer_call_fn_65445Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
в2Я
L__inference_transformer_block_layer_call_and_return_conditional_losses_66597
L__inference_transformer_block_layer_call_and_return_conditional_losses_66898А
ЇВЃ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
1__inference_transformer_block_layer_call_fn_66976
1__inference_transformer_block_layer_call_fn_66937А
ЇВЃ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
п2м
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66982
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66993Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Љ2І
8__inference_global_average_pooling1d_layer_call_fn_66987
8__inference_global_average_pooling1d_layer_call_fn_66998Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ц2У
D__inference_dropout_2_layer_call_and_return_conditional_losses_67015
D__inference_dropout_2_layer_call_and_return_conditional_losses_67010Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_dropout_2_layer_call_fn_67020
)__inference_dropout_2_layer_call_fn_67025Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_dense_6_layer_call_and_return_conditional_losses_67035Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_6_layer_call_fn_67044Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_64309Ц
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!џџџџџџџџџџџџџџџџџџ
ї2є
)__inference_p_re_lu_1_layer_call_fn_64317Ц
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *&Ђ#
!џџџџџџџџџџџџџџџџџџ
Ц2У
D__inference_dropout_3_layer_call_and_return_conditional_losses_67061
D__inference_dropout_3_layer_call_and_return_conditional_losses_67056Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
)__inference_dropout_3_layer_call_fn_67071
)__inference_dropout_3_layer_call_fn_67066Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_dense_7_layer_call_and_return_conditional_losses_67082Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_7_layer_call_fn_67091Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2B0
#__inference_signature_wrapper_65504input_1
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_67155
E__inference_sequential_layer_call_and_return_conditional_losses_67219
E__inference_sequential_layer_call_and_return_conditional_losses_64198
E__inference_sequential_layer_call_and_return_conditional_losses_64214Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
*__inference_sequential_layer_call_fn_64246
*__inference_sequential_layer_call_fn_67249
*__inference_sequential_layer_call_fn_64277
*__inference_sequential_layer_call_fn_67234Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
К2ЗД
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ј2ЅЂ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_4_layer_call_and_return_conditional_losses_67287Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_4_layer_call_fn_67298Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_5_layer_call_and_return_conditional_losses_67328Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_5_layer_call_fn_67337Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
B__inference_p_re_lu_layer_call_and_return_conditional_losses_64081г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
2џ
'__inference_p_re_lu_layer_call_fn_64089г
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *3Ђ0
.+'џџџџџџџџџџџџџџџџџџџџџџџџџџџЇ
 __inference__wrapped_model_64068;<=>?@ABHICDEFGJK!"'015Ђ2
+Ђ(
&#
input_1џџџџџџџџџ2Х
Њ "1Њ.
,
dense_7!
dense_7џџџџџџџџџ	­
B__inference_dense_4_layer_call_and_return_conditional_losses_67287gCDE4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ2Х
Њ "*Ђ'
 
0џџџџџџџџџ2
 
'__inference_dense_4_layer_call_fn_67298ZCDE4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ2Х
Њ "џџџџџџџџџ2Ќ
B__inference_dense_5_layer_call_and_return_conditional_losses_67328fFG4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ2
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 
'__inference_dense_5_layer_call_fn_67337YFG4Ђ1
*Ђ'
%"
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ2ХЄ
B__inference_dense_6_layer_call_and_return_conditional_losses_67035^!"0Ђ-
&Ђ#
!
inputsџџџџџџџџџХ
Њ "&Ђ#

0џџџџџџџџџ
 |
'__inference_dense_6_layer_call_fn_67044Q!"0Ђ-
&Ђ#
!
inputsџџџџџџџџџХ
Њ "џџџџџџџџџЃ
B__inference_dense_7_layer_call_and_return_conditional_losses_67082]010Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ	
 {
'__inference_dense_7_layer_call_fn_67091P010Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ	І
D__inference_dropout_2_layer_call_and_return_conditional_losses_67010^4Ђ1
*Ђ'
!
inputsџџџџџџџџџХ
p
Њ "&Ђ#

0џџџџџџџџџХ
 І
D__inference_dropout_2_layer_call_and_return_conditional_losses_67015^4Ђ1
*Ђ'
!
inputsџџџџџџџџџХ
p 
Њ "&Ђ#

0џџџџџџџџџХ
 ~
)__inference_dropout_2_layer_call_fn_67020Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџХ
p
Њ "џџџџџџџџџХ~
)__inference_dropout_2_layer_call_fn_67025Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџХ
p 
Њ "џџџџџџџџџХІ
D__inference_dropout_3_layer_call_and_return_conditional_losses_67056^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 І
D__inference_dropout_3_layer_call_and_return_conditional_losses_67061^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 ~
)__inference_dropout_3_layer_call_fn_67066Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ~
)__inference_dropout_3_layer_call_fn_67071Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџЩ
G__inference_functional_1_layer_call_and_return_conditional_losses_65181~;<=>?@ABHICDEFGJK!"'01=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ2Х
p

 
Њ "%Ђ"

0џџџџџџџџџ	
 Щ
G__inference_functional_1_layer_call_and_return_conditional_losses_65236~;<=>?@ABHICDEFGJK!"'01=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ2Х
p 

 
Њ "%Ђ"

0џџџџџџџџџ	
 Ш
G__inference_functional_1_layer_call_and_return_conditional_losses_65858};<=>?@ABHICDEFGJK!"'01<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p

 
Њ "%Ђ"

0џџџџџџџџџ	
 Ш
G__inference_functional_1_layer_call_and_return_conditional_losses_66184};<=>?@ABHICDEFGJK!"'01<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p 

 
Њ "%Ђ"

0џџџџџџџџџ	
 Ё
,__inference_functional_1_layer_call_fn_65341q;<=>?@ABHICDEFGJK!"'01=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ2Х
p

 
Њ "џџџџџџџџџ	Ё
,__inference_functional_1_layer_call_fn_65445q;<=>?@ABHICDEFGJK!"'01=Ђ:
3Ђ0
&#
input_1џџџџџџџџџ2Х
p 

 
Њ "џџџџџџџџџ	 
,__inference_functional_1_layer_call_fn_66233p;<=>?@ABHICDEFGJK!"'01<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p

 
Њ "џџџџџџџџџ	 
,__inference_functional_1_layer_call_fn_66282p;<=>?@ABHICDEFGJK!"'01<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p 

 
Њ "џџџџџџџџџ	Й
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66982b8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х

 
Њ "&Ђ#

0џџџџџџџџџХ
 в
S__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_66993{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 
8__inference_global_average_pooling1d_layer_call_fn_66987U8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х

 
Њ "џџџџџџџџџХЊ
8__inference_global_average_pooling1d_layer_call_fn_66998nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ­
D__inference_p_re_lu_1_layer_call_and_return_conditional_losses_64309e'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 
)__inference_p_re_lu_1_layer_call_fn_64317X'8Ђ5
.Ђ+
)&
inputsџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџМ
B__inference_p_re_lu_layer_call_and_return_conditional_losses_64081vEEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "*Ђ'
 
0џџџџџџџџџ2
 
'__inference_p_re_lu_layer_call_fn_64089iEEЂB
;Ђ8
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "џџџџџџџџџ2С
E__inference_sequential_layer_call_and_return_conditional_losses_64198xCDEFGCЂ@
9Ђ6
,)
dense_4_inputџџџџџџџџџ2Х
p

 
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 С
E__inference_sequential_layer_call_and_return_conditional_losses_64214xCDEFGCЂ@
9Ђ6
,)
dense_4_inputџџџџџџџџџ2Х
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 К
E__inference_sequential_layer_call_and_return_conditional_losses_67155qCDEFG<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p

 
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 К
E__inference_sequential_layer_call_and_return_conditional_losses_67219qCDEFG<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p 

 
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 
*__inference_sequential_layer_call_fn_64246kCDEFGCЂ@
9Ђ6
,)
dense_4_inputџџџџџџџџџ2Х
p

 
Њ "џџџџџџџџџ2Х
*__inference_sequential_layer_call_fn_64277kCDEFGCЂ@
9Ђ6
,)
dense_4_inputџџџџџџџџџ2Х
p 

 
Њ "џџџџџџџџџ2Х
*__inference_sequential_layer_call_fn_67234dCDEFG<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p

 
Њ "џџџџџџџџџ2Х
*__inference_sequential_layer_call_fn_67249dCDEFG<Ђ9
2Ђ/
%"
inputsџџџџџџџџџ2Х
p 

 
Њ "џџџџџџџџџ2ХЕ
#__inference_signature_wrapper_65504;<=>?@ABHICDEFGJK!"'01@Ђ=
Ђ 
6Њ3
1
input_1&#
input_1џџџџџџџџџ2Х"1Њ.
,
dense_7!
dense_7џџџџџџџџџ	Щ
L__inference_transformer_block_layer_call_and_return_conditional_losses_66597y;<=>?@ABHICDEFGJK8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х
p
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 Щ
L__inference_transformer_block_layer_call_and_return_conditional_losses_66898y;<=>?@ABHICDEFGJK8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х
p 
Њ "*Ђ'
 
0џџџџџџџџџ2Х
 Ё
1__inference_transformer_block_layer_call_fn_66937l;<=>?@ABHICDEFGJK8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х
p
Њ "џџџџџџџџџ2ХЁ
1__inference_transformer_block_layer_call_fn_66976l;<=>?@ABHICDEFGJK8Ђ5
.Ђ+
%"
inputsџџџџџџџџџ2Х
p 
Њ "џџџџџџџџџ2Х