??<
?/?/
:
Add
x"T
y"T
z"T"
Ttype:
2	
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint?
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	
?
)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNorm
x"T

scale"T
offset"T	
mean"T
variance"T
y"T

batch_mean"T
batch_variance"T
reserve_space_1"T
reserve_space_2"T"
Ttype:
2"
epsilonfloat%??8"-
data_formatstringNHWC:
NHWCNCHW"
is_trainingbool(
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z
?
!
LoopCond	
input


output

?
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
q
ResizeBilinear
images"T
size
resized_images"
Ttype:
2
	"
align_cornersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
+
Rint
x"T
y"T"
Ttype:
2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:?
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype?
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype?
9
TensorArraySizeV3

handle
flow_in
size?
?
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring ?
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype?
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?"serve*
1.12.0-rc22unknownۨ0

global_step/Initializer/zerosConst*
_class
loc:@global_step*
value	B	 R *
dtype0	*
_output_shapes
: 
?
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shared_name *
_class
loc:@global_step*
	container *
shape: 
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
?
imagePlaceholder*
dtype0*A
_output_shapes/
-:+???????????????????????????*6
shape-:+???????????????????????????
s
true_image_shapePlaceholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
?
ToFloatCastimage*
Truncate( *A
_output_shapes/
-:+???????????????????????????*

DstT0*

SrcT0
v
map/ToFloatCasttrue_image_shape*
Truncate( *'
_output_shapes
:?????????*

DstT0*

SrcT0
P
	map/ShapeShapeToFloat*
_output_shapes
:*
T0*
out_type0
a
map/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
map/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
map/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_sliceStridedSlice	map/Shapemap/strided_slice/stackmap/strided_slice/stack_1map/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
?
map/TensorArrayTensorArrayV3map/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name 
?
map/TensorArray_1TensorArrayV3map/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name 
c
map/TensorArrayUnstack/ShapeShapeToFloat*
_output_shapes
:*
T0*
out_type0
t
*map/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,map/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
v
,map/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
$map/TensorArrayUnstack/strided_sliceStridedSlicemap/TensorArrayUnstack/Shape*map/TensorArrayUnstack/strided_slice/stack,map/TensorArrayUnstack/strided_slice/stack_1,map/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
"map/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"map/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayUnstack/rangeRange"map/TensorArrayUnstack/range/start$map/TensorArrayUnstack/strided_slice"map/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????*

Tidx0
?
>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArraymap/TensorArrayUnstack/rangeToFloatmap/TensorArray:1*
_output_shapes
: *
T0*
_class
loc:@ToFloat
i
map/TensorArrayUnstack_1/ShapeShapemap/ToFloat*
_output_shapes
:*
T0*
out_type0
v
,map/TensorArrayUnstack_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.map/TensorArrayUnstack_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.map/TensorArrayUnstack_1/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
&map/TensorArrayUnstack_1/strided_sliceStridedSlicemap/TensorArrayUnstack_1/Shape,map/TensorArrayUnstack_1/strided_slice/stack.map/TensorArrayUnstack_1/strided_slice/stack_1.map/TensorArrayUnstack_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
f
$map/TensorArrayUnstack_1/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$map/TensorArrayUnstack_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayUnstack_1/rangeRange$map/TensorArrayUnstack_1/range/start&map/TensorArrayUnstack_1/strided_slice$map/TensorArrayUnstack_1/range/delta*#
_output_shapes
:?????????*

Tidx0
?
@map/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArray_1map/TensorArrayUnstack_1/rangemap/ToFloatmap/TensorArray_1:1*
T0*
_class
loc:@map/ToFloat*
_output_shapes
: 
K
	map/ConstConst*
dtype0*
_output_shapes
: *
value	B : 
?
map/TensorArray_2TensorArrayV3map/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes( *
tensor_array_name *
dtype0*
_output_shapes

:: 
?
map/TensorArray_3TensorArrayV3map/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes( *
tensor_array_name *
dtype0*
_output_shapes

:: 
?
map/TensorArray_4TensorArrayV3map/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes( 
?
map/TensorArray_5TensorArrayV3map/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes( 
?
map/TensorArray_6TensorArrayV3map/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
?
map/TensorArray_7TensorArrayV3map/strided_slice*
tensor_array_name *
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(
?
map/TensorArray_8TensorArrayV3map/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name 
?
map/TensorArray_9TensorArrayV3map/strided_slice*
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
?
map/TensorArrayReadV3TensorArrayReadV3map/TensorArray	map/Const>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
dtype0*4
_output_shapes"
 :??????????????????
?
map/TensorArrayReadV3_1TensorArrayReadV3map/TensorArray_1	map/Const@map/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3*
dtype0*
_output_shapes
:
m
map/CastCastmap/TensorArrayReadV3_1*
Truncate( *
_output_shapes
:*

DstT0*

SrcT0
^
	map/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
c
map/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
e
map/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_1StridedSlicemap/Castmap/strided_slice_1/stackmap/strided_slice_1/stack_1map/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
c
map/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_2StridedSlicemap/Castmap/strided_slice_2/stackmap/strided_slice_2/stack_1map/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
O
map/stack_1/2Const*
dtype0*
_output_shapes
: *
value	B :
?
map/stack_1Packmap/strided_slice_1map/strided_slice_2map/stack_1/2*
N*
_output_shapes
:*
T0*

axis 
?
	map/SliceSlicemap/TensorArrayReadV3	map/stackmap/stack_1*
Index0*
T0*4
_output_shapes"
 :??????????????????
N
map/Const_1Const*
dtype0*
_output_shapes
: *
value
B :?
T
map/Shape_1Shape	map/Slice*
_output_shapes
:*
T0*
out_type0
c
map/strided_slice_3/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
map/strided_slice_3/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_3StridedSlicemap/Shape_1map/strided_slice_3/stackmap/strided_slice_3/stack_1map/strided_slice_3/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
c
map/strided_slice_4/stackConst*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_4/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_4StridedSlicemap/Shape_1map/strided_slice_4/stackmap/strided_slice_4/stack_1map/strided_slice_4/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
j
map/ToFloat_1Castmap/strided_slice_3*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
j
map/ToFloat_2Castmap/strided_slice_4*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
b
map/ToFloat_3Castmap/Const_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
U
map/GreaterGreatermap/ToFloat_1map/ToFloat_2*
_output_shapes
: *
T0
V
map/cond/SwitchSwitchmap/Greatermap/Greater*
_output_shapes
: : *
T0

Q
map/cond/switch_tIdentitymap/cond/Switch:1*
_output_shapes
: *
T0

O
map/cond/switch_fIdentitymap/cond/Switch*
T0
*
_output_shapes
: 
J
map/cond/pred_idIdentitymap/Greater*
T0
*
_output_shapes
: 
t
map/cond/truedivRealDivmap/cond/truediv/Switch:1map/cond/truediv/Switch_1:1*
T0*
_output_shapes
: 
?
map/cond/truediv/SwitchSwitchmap/ToFloat_3map/cond/pred_id*
T0* 
_class
loc:@map/ToFloat_3*
_output_shapes
: : 
?
map/cond/truediv/Switch_1Switchmap/ToFloat_2map/cond/pred_id*
T0* 
_class
loc:@map/ToFloat_2*
_output_shapes
: : 
v
map/cond/truediv_1RealDivmap/cond/truediv_1/Switchmap/cond/truediv_1/Switch_1*
T0*
_output_shapes
: 
?
map/cond/truediv_1/SwitchSwitchmap/ToFloat_3map/cond/pred_id*
T0* 
_class
loc:@map/ToFloat_3*
_output_shapes
: : 
?
map/cond/truediv_1/Switch_1Switchmap/ToFloat_1map/cond/pred_id*
T0* 
_class
loc:@map/ToFloat_1*
_output_shapes
: : 
i
map/cond/MergeMergemap/cond/truediv_1map/cond/truediv*
T0*
N*
_output_shapes
: : 
N
map/mulMulmap/ToFloat_1map/cond/Merge*
_output_shapes
: *
T0
:
map/RintRintmap/mul*
T0*
_output_shapes
: 
]
map/ToInt32Castmap/Rint*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
P
	map/mul_1Mulmap/ToFloat_2map/cond/Merge*
T0*
_output_shapes
: 
>

map/Rint_1Rint	map/mul_1*
T0*
_output_shapes
: 
a
map/ToInt32_1Cast
map/Rint_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
T
map/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B : 
?
map/ExpandDims
ExpandDims	map/Slicemap/ExpandDims/dim*8
_output_shapes&
$:"??????????????????*

Tdim0*
T0
u
map/ResizeBilinear/sizePackmap/ToInt32map/ToInt32_1*
T0*

axis *
N*
_output_shapes
:
?
map/ResizeBilinearResizeBilinearmap/ExpandDimsmap/ResizeBilinear/size*8
_output_shapes&
$:"??????????????????*
align_corners( *
T0
}
map/SqueezeSqueezemap/ResizeBilinear*4
_output_shapes"
 :??????????????????*
squeeze_dims
 *
T0
V
map/Shape_2Shapemap/Squeeze*
T0*
out_type0*
_output_shapes
:
c
map/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB: 
e
map/strided_slice_5/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
e
map/strided_slice_5/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_5StridedSlicemap/Shape_2map/strided_slice_5/stackmap/strided_slice_5/stack_1map/strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
V
map/Shape_3Shapemap/Squeeze*
T0*
out_type0*
_output_shapes
:
c
map/strided_slice_6/stackConst*
dtype0*
_output_shapes
:*
valueB:
e
map/strided_slice_6/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_6StridedSlicemap/Shape_3map/strided_slice_6/stackmap/strided_slice_6/stack_1map/strided_slice_6/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
L
	map/sub/yConst*
dtype0*
_output_shapes
: *
value
B :?
O
map/subSubmap/strided_slice_5	map/sub/y*
_output_shapes
: *
T0
O
map/truediv/yConst*
dtype0*
_output_shapes
: *
value	B :
a
map/truediv/CastCastmap/sub*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
i
map/truediv/Cast_1Castmap/truediv/y*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
]
map/truedivRealDivmap/truediv/Castmap/truediv/Cast_1*
_output_shapes
: *
T0
N
map/sub_1/yConst*
dtype0*
_output_shapes
: *
value
B :?
S
	map/sub_1Submap/strided_slice_6map/sub_1/y*
T0*
_output_shapes
: 
Q
map/truediv_1/yConst*
dtype0*
_output_shapes
: *
value	B :
e
map/truediv_1/CastCast	map/sub_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
m
map/truediv_1/Cast_1Castmap/truediv_1/y*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
c
map/truediv_1RealDivmap/truediv_1/Castmap/truediv_1/Cast_1*
T0*
_output_shapes
: 
V
map/Shape_4Shapemap/Squeeze*
T0*
out_type0*
_output_shapes
:
J
map/RankConst*
dtype0*
_output_shapes
: *
value	B :
M
map/Equal/yConst*
value	B :*
dtype0*
_output_shapes
: 
J
	map/EqualEqualmap/Rankmap/Equal/y*
_output_shapes
: *
T0
r
map/Assert/ConstConst*2
value)B' B!Rank of image must be equal to 3.*
dtype0*
_output_shapes
: 
z
map/Assert/Assert/data_0Const*
dtype0*
_output_shapes
: *2
value)B' B!Rank of image must be equal to 3.
]
map/Assert/AssertAssert	map/Equalmap/Assert/Assert/data_0*

T
2*
	summarize
w
map/strided_slice_7/stackConst^map/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
y
map/strided_slice_7/stack_1Const^map/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
y
map/strided_slice_7/stack_2Const^map/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
?
map/strided_slice_7StridedSlicemap/Shape_4map/strided_slice_7/stackmap/strided_slice_7/stack_1map/strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
map/stack_2/0Const^map/Assert/Assert*
dtype0*
_output_shapes
: *
value
B :?
d
map/stack_2/1Const^map/Assert/Assert*
value
B :?*
dtype0*
_output_shapes
: 
?
map/stack_2Packmap/stack_2/0map/stack_2/1map/strided_slice_7*
N*
_output_shapes
:*
T0*

axis 
c
map/strided_slice_8/stackConst*
dtype0*
_output_shapes
:*
valueB: 
e
map/strided_slice_8/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
e
map/strided_slice_8/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
map/strided_slice_8StridedSlicemap/Shape_4map/strided_slice_8/stackmap/strided_slice_8/stack_1map/strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
U
map/GreaterEqual/yConst*
dtype0*
_output_shapes
: *
value
B :?
j
map/GreaterEqualGreaterEqualmap/strided_slice_8map/GreaterEqual/y*
_output_shapes
: *
T0
c
map/strided_slice_9/stackConst*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_9/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
map/strided_slice_9/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
map/strided_slice_9StridedSlicemap/Shape_4map/strided_slice_9/stackmap/strided_slice_9/stack_1map/strided_slice_9/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
W
map/GreaterEqual_1/yConst*
dtype0*
_output_shapes
: *
value
B :?
n
map/GreaterEqual_1GreaterEqualmap/strided_slice_9map/GreaterEqual_1/y*
_output_shapes
: *
T0
Z
map/LogicalAnd
LogicalAndmap/GreaterEqualmap/GreaterEqual_1*
_output_shapes
: 
y
map/Assert_1/ConstConst*
dtype0*
_output_shapes
: *7
value.B, B&Crop size greater than the image size.
?
map/Assert_1/Assert/data_0Const*
dtype0*
_output_shapes
: *7
value.B, B&Crop size greater than the image size.
f
map/Assert_1/AssertAssertmap/LogicalAndmap/Assert_1/Assert/data_0*

T
2*
	summarize
V
map/stack_3/2Const*
valueB 2        *
dtype0*
_output_shapes
: 
x
map/stack_3Packmap/truedivmap/truediv_1map/stack_3/2*
N*
_output_shapes
:*
T0*

axis 
f
map/ToInt32_2Castmap/stack_3*
Truncate( *
_output_shapes
:*

DstT0*

SrcT0
?
map/Slice_1Slicemap/Squeezemap/ToInt32_2map/stack_2^map/Assert_1/Assert*
Index0*
T0*-
_output_shapes
:???????????
m
map/ReshapeReshapemap/Slice_1map/stack_2*
T0*
Tshape0*$
_output_shapes
:??
M
map/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
U
map/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
?
	map/splitSplitmap/split/split_dimmap/Reshape*
T0*D
_output_shapes2
0:??:??:??*
	num_split
P
map/sub_2/yConst*
valueB
 *)\?B*
dtype0*
_output_shapes
: 
W
	map/sub_2Sub	map/splitmap/sub_2/y*$
_output_shapes
:??*
T0
P
map/sub_3/yConst*
valueB
 *\??B*
dtype0*
_output_shapes
: 
Y
	map/sub_3Submap/split:1map/sub_3/y*
T0*$
_output_shapes
:??
P
map/sub_4/yConst*
dtype0*
_output_shapes
: *
valueB
 *H??B
Y
	map/sub_4Submap/split:2map/sub_4/y*$
_output_shapes
:??*
T0
Q
map/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
?

map/concatConcatV2	map/sub_2	map/sub_3	map/sub_4map/concat/axis*
T0*
N*$
_output_shapes
:??*

Tidx0
`
map/Shape_5Const*!
valueB"?   ?      *
dtype0*
_output_shapes
:
c
map/ToFloat_4Castmap/Cast*
Truncate( *
_output_shapes
:*

DstT0*

SrcT0
f
map/ToFloat_5Castmap/Shape_5*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
K
	map/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
E
map/addAdd	map/Const	map/add/y*
T0*
_output_shapes
: 
?
'map/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3map/TensorArray_2	map/Const
map/concatmap/TensorArray_2:1*
_output_shapes
: *
T0*
_class
loc:@map/concat
?
)map/TensorArrayWrite_1/TensorArrayWriteV3TensorArrayWriteV3map/TensorArray_3	map/Const	map/Slicemap/TensorArray_3:1*
T0*
_class
loc:@map/Slice*
_output_shapes
: 
?
)map/TensorArrayWrite_2/TensorArrayWriteV3TensorArrayWriteV3map/TensorArray_4	map/Constmap/ToFloat_4map/TensorArray_4:1*
_output_shapes
: *
T0* 
_class
loc:@map/ToFloat_4
?
)map/TensorArrayWrite_3/TensorArrayWriteV3TensorArrayWriteV3map/TensorArray_5	map/Constmap/ToFloat_5map/TensorArray_5:1*
T0* 
_class
loc:@map/ToFloat_5*
_output_shapes
: 
`
map/Shape_6Const*
dtype0*
_output_shapes
:*!
valueB"?   ?      
T
map/Shape_7Shape	map/Slice*
T0*
out_type0*
_output_shapes
:
U
map/Shape_8Const*
valueB:*
dtype0*
_output_shapes
:
U
map/Shape_9Const*
valueB:*
dtype0*
_output_shapes
:
]
map/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
?
map/while/EnterEntermap/while/iteration_counter*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
?
map/while/Enter_1Entermap/add*
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context*
T0*
is_constant( 
?
map/while/Enter_2Enter'map/TensorArrayWrite/TensorArrayWriteV3*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
?
map/while/Enter_3Enter)map/TensorArrayWrite_1/TensorArrayWriteV3*
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context*
T0*
is_constant( 
?
map/while/Enter_4Enter)map/TensorArrayWrite_2/TensorArrayWriteV3*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
?
map/while/Enter_5Enter)map/TensorArrayWrite_3/TensorArrayWriteV3*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
?
map/while/Enter_6Entermap/Shape_6*
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant( 
?
map/while/Enter_7Entermap/Shape_7*
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant( 
?
map/while/Enter_8Entermap/Shape_8*
T0*
is_constant( *
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context
?
map/while/Enter_9Entermap/Shape_9*
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant( 
n
map/while/MergeMergemap/while/Entermap/while/NextIteration*
N*
_output_shapes
: : *
T0
t
map/while/Merge_1Mergemap/while/Enter_1map/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_2Mergemap/while/Enter_2map/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_3Mergemap/while/Enter_3map/while/NextIteration_3*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_4Mergemap/while/Enter_4map/while/NextIteration_4*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_5Mergemap/while/Enter_5map/while/NextIteration_5*
T0*
N*
_output_shapes
: : 
x
map/while/Merge_6Mergemap/while/Enter_6map/while/NextIteration_6*
T0*
N*
_output_shapes

:: 
x
map/while/Merge_7Mergemap/while/Enter_7map/while/NextIteration_7*
N*
_output_shapes

:: *
T0
x
map/while/Merge_8Mergemap/while/Enter_8map/while/NextIteration_8*
N*
_output_shapes

:: *
T0
x
map/while/Merge_9Mergemap/while/Enter_9map/while/NextIteration_9*
T0*
N*
_output_shapes

:: 
^
map/while/LessLessmap/while/Mergemap/while/Less/Enter*
_output_shapes
: *
T0
?
map/while/Less/EnterEntermap/strided_slice*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
b
map/while/Less_1Lessmap/while/Merge_1map/while/Less/Enter*
_output_shapes
: *
T0
\
map/while/LogicalAnd
LogicalAndmap/while/Lessmap/while/Less_1*
_output_shapes
: 
L
map/while/LoopCondLoopCondmap/while/LogicalAnd*
_output_shapes
: 
?
map/while/SwitchSwitchmap/while/Mergemap/while/LoopCond*
T0*"
_class
loc:@map/while/Merge*
_output_shapes
: : 
?
map/while/Switch_1Switchmap/while/Merge_1map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_1*
_output_shapes
: : 
?
map/while/Switch_2Switchmap/while/Merge_2map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_2*
_output_shapes
: : 
?
map/while/Switch_3Switchmap/while/Merge_3map/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map/while/Merge_3
?
map/while/Switch_4Switchmap/while/Merge_4map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_4*
_output_shapes
: : 
?
map/while/Switch_5Switchmap/while/Merge_5map/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map/while/Merge_5
?
map/while/Switch_6Switchmap/while/Merge_6map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_6* 
_output_shapes
::
?
map/while/Switch_7Switchmap/while/Merge_7map/while/LoopCond* 
_output_shapes
::*
T0*$
_class
loc:@map/while/Merge_7
?
map/while/Switch_8Switchmap/while/Merge_8map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_8* 
_output_shapes
::
?
map/while/Switch_9Switchmap/while/Merge_9map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_9* 
_output_shapes
::
S
map/while/IdentityIdentitymap/while/Switch:1*
_output_shapes
: *
T0
W
map/while/Identity_1Identitymap/while/Switch_1:1*
_output_shapes
: *
T0
W
map/while/Identity_2Identitymap/while/Switch_2:1*
_output_shapes
: *
T0
W
map/while/Identity_3Identitymap/while/Switch_3:1*
_output_shapes
: *
T0
W
map/while/Identity_4Identitymap/while/Switch_4:1*
_output_shapes
: *
T0
W
map/while/Identity_5Identitymap/while/Switch_5:1*
T0*
_output_shapes
: 
[
map/while/Identity_6Identitymap/while/Switch_6:1*
_output_shapes
:*
T0
[
map/while/Identity_7Identitymap/while/Switch_7:1*
T0*
_output_shapes
:
[
map/while/Identity_8Identitymap/while/Switch_8:1*
_output_shapes
:*
T0
[
map/while/Identity_9Identitymap/while/Switch_9:1*
T0*
_output_shapes
:
f
map/while/add/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
Z
map/while/addAddmap/while/Identitymap/while/add/y*
T0*
_output_shapes
: 
?
map/while/TensorArrayReadV3TensorArrayReadV3!map/while/TensorArrayReadV3/Entermap/while/Identity_1#map/while/TensorArrayReadV3/Enter_1*
dtype0*4
_output_shapes"
 :??????????????????
?
!map/while/TensorArrayReadV3/EnterEntermap/TensorArray*
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant(
?
#map/while/TensorArrayReadV3/Enter_1Enter>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
?
map/while/TensorArrayReadV3_1TensorArrayReadV3#map/while/TensorArrayReadV3_1/Entermap/while/Identity_1%map/while/TensorArrayReadV3_1/Enter_1*
dtype0*
_output_shapes
:
?
#map/while/TensorArrayReadV3_1/EnterEntermap/TensorArray_1*
parallel_iterations*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
is_constant(
?
%map/while/TensorArrayReadV3_1/Enter_1Enter@map/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *'

frame_namemap/while/while_context
y
map/while/CastCastmap/while/TensorArrayReadV3_1*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
y
map/while/stackConst^map/while/Identity*!
valueB"            *
dtype0*
_output_shapes
:
|
map/while/strided_slice/stackConst^map/while/Identity*
valueB: *
dtype0*
_output_shapes
:
~
map/while/strided_slice/stack_1Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
~
map/while/strided_slice/stack_2Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_sliceStridedSlicemap/while/Castmap/while/strided_slice/stackmap/while/strided_slice/stack_1map/while/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
~
map/while/strided_slice_1/stackConst^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
!map/while/strided_slice_1/stack_1Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_1/stack_2Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while/strided_slice_1StridedSlicemap/while/Castmap/while/strided_slice_1/stack!map/while/strided_slice_1/stack_1!map/while/strided_slice_1/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask 
j
map/while/stack_1/2Const^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
?
map/while/stack_1Packmap/while/strided_slicemap/while/strided_slice_1map/while/stack_1/2*
T0*

axis *
N*
_output_shapes
:
?
map/while/SliceSlicemap/while/TensorArrayReadV3map/while/stackmap/while/stack_1*4
_output_shapes"
 :??????????????????*
Index0*
T0
g
map/while/ConstConst^map/while/Identity*
value
B :?*
dtype0*
_output_shapes
: 
^
map/while/ShapeShapemap/while/Slice*
_output_shapes
:*
T0*
out_type0
~
map/while/strided_slice_2/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
!map/while/strided_slice_2/stack_1Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
!map/while/strided_slice_2/stack_2Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while/strided_slice_2StridedSlicemap/while/Shapemap/while/strided_slice_2/stack!map/while/strided_slice_2/stack_1!map/while/strided_slice_2/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
~
map/while/strided_slice_3/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_3/stack_1Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_3/stack_2Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_slice_3StridedSlicemap/while/Shapemap/while/strided_slice_3/stack!map/while/strided_slice_3/stack_1!map/while/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
t
map/while/ToFloatCastmap/while/strided_slice_2*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
v
map/while/ToFloat_1Castmap/while/strided_slice_3*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
l
map/while/ToFloat_2Castmap/while/Const*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
e
map/while/GreaterGreatermap/while/ToFloatmap/while/ToFloat_1*
_output_shapes
: *
T0
h
map/while/cond/SwitchSwitchmap/while/Greatermap/while/Greater*
_output_shapes
: : *
T0

]
map/while/cond/switch_tIdentitymap/while/cond/Switch:1*
_output_shapes
: *
T0

[
map/while/cond/switch_fIdentitymap/while/cond/Switch*
_output_shapes
: *
T0

V
map/while/cond/pred_idIdentitymap/while/Greater*
T0
*
_output_shapes
: 
?
map/while/cond/truedivRealDivmap/while/cond/truediv/Switch:1!map/while/cond/truediv/Switch_1:1*
_output_shapes
: *
T0
?
map/while/cond/truediv/SwitchSwitchmap/while/ToFloat_2map/while/cond/pred_id*
_output_shapes
: : *
T0*&
_class
loc:@map/while/ToFloat_2
?
map/while/cond/truediv/Switch_1Switchmap/while/ToFloat_1map/while/cond/pred_id*
T0*&
_class
loc:@map/while/ToFloat_1*
_output_shapes
: : 
?
map/while/cond/truediv_1RealDivmap/while/cond/truediv_1/Switch!map/while/cond/truediv_1/Switch_1*
T0*
_output_shapes
: 
?
map/while/cond/truediv_1/SwitchSwitchmap/while/ToFloat_2map/while/cond/pred_id*
_output_shapes
: : *
T0*&
_class
loc:@map/while/ToFloat_2
?
!map/while/cond/truediv_1/Switch_1Switchmap/while/ToFloatmap/while/cond/pred_id*
_output_shapes
: : *
T0*$
_class
loc:@map/while/ToFloat
{
map/while/cond/MergeMergemap/while/cond/truediv_1map/while/cond/truediv*
T0*
N*
_output_shapes
: : 
^
map/while/mulMulmap/while/ToFloatmap/while/cond/Merge*
_output_shapes
: *
T0
F
map/while/RintRintmap/while/mul*
_output_shapes
: *
T0
i
map/while/ToInt32Castmap/while/Rint*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
b
map/while/mul_1Mulmap/while/ToFloat_1map/while/cond/Merge*
T0*
_output_shapes
: 
J
map/while/Rint_1Rintmap/while/mul_1*
_output_shapes
: *
T0
m
map/while/ToInt32_1Castmap/while/Rint_1*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
o
map/while/ExpandDims/dimConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B : 
?
map/while/ExpandDims
ExpandDimsmap/while/Slicemap/while/ExpandDims/dim*

Tdim0*
T0*8
_output_shapes&
$:"??????????????????
?
map/while/ResizeBilinear/sizePackmap/while/ToInt32map/while/ToInt32_1*
N*
_output_shapes
:*
T0*

axis 
?
map/while/ResizeBilinearResizeBilinearmap/while/ExpandDimsmap/while/ResizeBilinear/size*
align_corners( *
T0*8
_output_shapes&
$:"??????????????????
?
map/while/SqueezeSqueezemap/while/ResizeBilinear*4
_output_shapes"
 :??????????????????*
squeeze_dims
 *
T0
b
map/while/Shape_1Shapemap/while/Squeeze*
T0*
out_type0*
_output_shapes
:
~
map/while/strided_slice_4/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
!map/while/strided_slice_4/stack_1Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
!map/while/strided_slice_4/stack_2Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_slice_4StridedSlicemap/while/Shape_1map/while/strided_slice_4/stack!map/while/strided_slice_4/stack_1!map/while/strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
b
map/while/Shape_2Shapemap/while/Squeeze*
T0*
out_type0*
_output_shapes
:
~
map/while/strided_slice_5/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_5/stack_1Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_5/stack_2Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_slice_5StridedSlicemap/while/Shape_2map/while/strided_slice_5/stack!map/while/strided_slice_5/stack_1!map/while/strided_slice_5/stack_2*
end_mask *
_output_shapes
: *
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
g
map/while/sub/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value
B :?
a
map/while/subSubmap/while/strided_slice_4map/while/sub/y*
T0*
_output_shapes
: 
j
map/while/truediv/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
m
map/while/truediv/CastCastmap/while/sub*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
u
map/while/truediv/Cast_1Castmap/while/truediv/y*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
o
map/while/truedivRealDivmap/while/truediv/Castmap/while/truediv/Cast_1*
_output_shapes
: *
T0
i
map/while/sub_1/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value
B :?
e
map/while/sub_1Submap/while/strided_slice_5map/while/sub_1/y*
T0*
_output_shapes
: 
l
map/while/truediv_1/yConst^map/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
q
map/while/truediv_1/CastCastmap/while/sub_1*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
y
map/while/truediv_1/Cast_1Castmap/while/truediv_1/y*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
u
map/while/truediv_1RealDivmap/while/truediv_1/Castmap/while/truediv_1/Cast_1*
_output_shapes
: *
T0
b
map/while/Shape_3Shapemap/while/Squeeze*
_output_shapes
:*
T0*
out_type0
e
map/while/RankConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
h
map/while/Equal/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
\
map/while/EqualEqualmap/while/Rankmap/while/Equal/y*
T0*
_output_shapes
: 
?
map/while/Assert/ConstConst^map/while/Identity*
dtype0*
_output_shapes
: *2
value)B' B!Rank of image must be equal to 3.
?
map/while/Assert/Assert/data_0Const^map/while/Identity*2
value)B' B!Rank of image must be equal to 3.*
dtype0*
_output_shapes
: 
o
map/while/Assert/AssertAssertmap/while/Equalmap/while/Assert/Assert/data_0*

T
2*
	summarize
?
map/while/strided_slice_6/stackConst^map/while/Assert/Assert*
valueB:*
dtype0*
_output_shapes
:
?
!map/while/strided_slice_6/stack_1Const^map/while/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_6/stack_2Const^map/while/Assert/Assert*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_slice_6StridedSlicemap/while/Shape_3map/while/strided_slice_6/stack!map/while/strided_slice_6/stack_1!map/while/strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
p
map/while/stack_2/0Const^map/while/Assert/Assert*
value
B :?*
dtype0*
_output_shapes
: 
p
map/while/stack_2/1Const^map/while/Assert/Assert*
dtype0*
_output_shapes
: *
value
B :?
?
map/while/stack_2Packmap/while/stack_2/0map/while/stack_2/1map/while/strided_slice_6*
T0*

axis *
N*
_output_shapes
:
~
map/while/strided_slice_7/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
!map/while/strided_slice_7/stack_1Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
!map/while/strided_slice_7/stack_2Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while/strided_slice_7StridedSlicemap/while/Shape_3map/while/strided_slice_7/stack!map/while/strided_slice_7/stack_1!map/while/strided_slice_7/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
p
map/while/GreaterEqual/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value
B :?
|
map/while/GreaterEqualGreaterEqualmap/while/strided_slice_7map/while/GreaterEqual/y*
_output_shapes
: *
T0
~
map/while/strided_slice_8/stackConst^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_8/stack_1Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
?
!map/while/strided_slice_8/stack_2Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while/strided_slice_8StridedSlicemap/while/Shape_3map/while/strided_slice_8/stack!map/while/strided_slice_8/stack_1!map/while/strided_slice_8/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
r
map/while/GreaterEqual_1/yConst^map/while/Identity*
value
B :?*
dtype0*
_output_shapes
: 
?
map/while/GreaterEqual_1GreaterEqualmap/while/strided_slice_8map/while/GreaterEqual_1/y*
T0*
_output_shapes
: 
n
map/while/LogicalAnd_1
LogicalAndmap/while/GreaterEqualmap/while/GreaterEqual_1*
_output_shapes
: 
?
map/while/Assert_1/ConstConst^map/while/Identity*7
value.B, B&Crop size greater than the image size.*
dtype0*
_output_shapes
: 
?
 map/while/Assert_1/Assert/data_0Const^map/while/Identity*
dtype0*
_output_shapes
: *7
value.B, B&Crop size greater than the image size.
z
map/while/Assert_1/AssertAssertmap/while/LogicalAnd_1 map/while/Assert_1/Assert/data_0*

T
2*
	summarize
q
map/while/stack_3/2Const^map/while/Identity*
dtype0*
_output_shapes
: *
valueB 2        
?
map/while/stack_3Packmap/while/truedivmap/while/truediv_1map/while/stack_3/2*
N*
_output_shapes
:*
T0*

axis 
r
map/while/ToInt32_2Castmap/while/stack_3*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
?
map/while/Slice_1Slicemap/while/Squeezemap/while/ToInt32_2map/while/stack_2^map/while/Assert_1/Assert*
Index0*
T0*-
_output_shapes
:???????????

map/while/ReshapeReshapemap/while/Slice_1map/while/stack_2*
T0*
Tshape0*$
_output_shapes
:??
h
map/while/Const_1Const^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
p
map/while/split/split_dimConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
?
map/while/splitSplitmap/while/split/split_dimmap/while/Reshape*D
_output_shapes2
0:??:??:??*
	num_split*
T0
k
map/while/sub_2/yConst^map/while/Identity*
valueB
 *)\?B*
dtype0*
_output_shapes
: 
i
map/while/sub_2Submap/while/splitmap/while/sub_2/y*
T0*$
_output_shapes
:??
k
map/while/sub_3/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *\??B
k
map/while/sub_3Submap/while/split:1map/while/sub_3/y*$
_output_shapes
:??*
T0
k
map/while/sub_4/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
valueB
 *H??B
k
map/while/sub_4Submap/while/split:2map/while/sub_4/y*$
_output_shapes
:??*
T0
l
map/while/concat/axisConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
?
map/while/concatConcatV2map/while/sub_2map/while/sub_3map/while/sub_4map/while/concat/axis*
T0*
N*$
_output_shapes
:??*

Tidx0
{
map/while/Shape_4Const^map/while/Identity*
dtype0*
_output_shapes
:*!
valueB"?   ?      
o
map/while/ToFloat_3Castmap/while/Cast*

SrcT0*
Truncate( *
_output_shapes
:*

DstT0
r
map/while/ToFloat_4Castmap/while/Shape_4*
Truncate( *
_output_shapes
:*

DstT0*

SrcT0
h
map/while/add_1/yConst^map/while/Identity*
dtype0*
_output_shapes
: *
value	B :
`
map/while/add_1Addmap/while/Identity_1map/while/add_1/y*
T0*
_output_shapes
: 
?
-map/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33map/while/TensorArrayWrite/TensorArrayWriteV3/Entermap/while/Identity_1map/while/concatmap/while/Identity_2*
_output_shapes
: *
T0*
_class
loc:@map/concat
?
3map/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_2*
is_constant(*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
_class
loc:@map/concat*
parallel_iterations
?
/map/while/TensorArrayWrite_1/TensorArrayWriteV3TensorArrayWriteV35map/while/TensorArrayWrite_1/TensorArrayWriteV3/Entermap/while/Identity_1map/while/Slicemap/while/Identity_3*
_output_shapes
: *
T0*
_class
loc:@map/Slice
?
5map/while/TensorArrayWrite_1/TensorArrayWriteV3/EnterEntermap/TensorArray_3*
parallel_iterations*
is_constant(*
_output_shapes
:*'

frame_namemap/while/while_context*
T0*
_class
loc:@map/Slice
?
/map/while/TensorArrayWrite_2/TensorArrayWriteV3TensorArrayWriteV35map/while/TensorArrayWrite_2/TensorArrayWriteV3/Entermap/while/Identity_1map/while/ToFloat_3map/while/Identity_4*
T0* 
_class
loc:@map/ToFloat_4*
_output_shapes
: 
?
5map/while/TensorArrayWrite_2/TensorArrayWriteV3/EnterEntermap/TensorArray_4*
_output_shapes
:*'

frame_namemap/while/while_context*
T0* 
_class
loc:@map/ToFloat_4*
parallel_iterations*
is_constant(
?
/map/while/TensorArrayWrite_3/TensorArrayWriteV3TensorArrayWriteV35map/while/TensorArrayWrite_3/TensorArrayWriteV3/Entermap/while/Identity_1map/while/ToFloat_4map/while/Identity_5*
T0* 
_class
loc:@map/ToFloat_5*
_output_shapes
: 
?
5map/while/TensorArrayWrite_3/TensorArrayWriteV3/EnterEntermap/TensorArray_5*
T0* 
_class
loc:@map/ToFloat_5*
parallel_iterations*
is_constant(*
_output_shapes
:*'

frame_namemap/while/while_context
{
map/while/Shape_5Const^map/while/Identity*!
valueB"?   ?      *
dtype0*
_output_shapes
:
j
map/while/MaximumMaximummap/while/Shape_5map/while/Identity_6*
_output_shapes
:*
T0
`
map/while/Shape_6Shapemap/while/Slice*
_output_shapes
:*
T0*
out_type0
l
map/while/Maximum_1Maximummap/while/Shape_6map/while/Identity_7*
_output_shapes
:*
T0
p
map/while/Shape_7Const^map/while/Identity*
dtype0*
_output_shapes
:*
valueB:
l
map/while/Maximum_2Maximummap/while/Shape_7map/while/Identity_8*
T0*
_output_shapes
:
p
map/while/Shape_8Const^map/while/Identity*
valueB:*
dtype0*
_output_shapes
:
l
map/while/Maximum_3Maximummap/while/Shape_8map/while/Identity_9*
_output_shapes
:*
T0
X
map/while/NextIterationNextIterationmap/while/add*
T0*
_output_shapes
: 
\
map/while/NextIteration_1NextIterationmap/while/add_1*
_output_shapes
: *
T0
z
map/while/NextIteration_2NextIteration-map/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
|
map/while/NextIteration_3NextIteration/map/while/TensorArrayWrite_1/TensorArrayWriteV3*
_output_shapes
: *
T0
|
map/while/NextIteration_4NextIteration/map/while/TensorArrayWrite_2/TensorArrayWriteV3*
_output_shapes
: *
T0
|
map/while/NextIteration_5NextIteration/map/while/TensorArrayWrite_3/TensorArrayWriteV3*
T0*
_output_shapes
: 
b
map/while/NextIteration_6NextIterationmap/while/Maximum*
_output_shapes
:*
T0
d
map/while/NextIteration_7NextIterationmap/while/Maximum_1*
_output_shapes
:*
T0
d
map/while/NextIteration_8NextIterationmap/while/Maximum_2*
_output_shapes
:*
T0
d
map/while/NextIteration_9NextIterationmap/while/Maximum_3*
T0*
_output_shapes
:
I
map/while/ExitExitmap/while/Switch*
_output_shapes
: *
T0
M
map/while/Exit_1Exitmap/while/Switch_1*
T0*
_output_shapes
: 
M
map/while/Exit_2Exitmap/while/Switch_2*
T0*
_output_shapes
: 
M
map/while/Exit_3Exitmap/while/Switch_3*
T0*
_output_shapes
: 
M
map/while/Exit_4Exitmap/while/Switch_4*
T0*
_output_shapes
: 
M
map/while/Exit_5Exitmap/while/Switch_5*
T0*
_output_shapes
: 
Q
map/while/Exit_6Exitmap/while/Switch_6*
_output_shapes
:*
T0
Q
map/while/Exit_7Exitmap/while/Switch_7*
T0*
_output_shapes
:
Q
map/while/Exit_8Exitmap/while/Switch_8*
_output_shapes
:*
T0
Q
map/while/Exit_9Exitmap/while/Switch_9*
_output_shapes
:*
T0
_
map/while_1/iteration_counterConst*
dtype0*
_output_shapes
: *
value	B : 
?
map/while_1/EnterEntermap/while_1/iteration_counter*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/Enter_1Enter	map/Const*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context*
T0*
is_constant( 
?
map/while_1/Enter_2Entermap/TensorArray_6:1*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/Enter_3Entermap/TensorArray_7:1*
T0*
is_constant( *
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/Enter_4Entermap/TensorArray_8:1*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context*
T0*
is_constant( 
?
map/while_1/Enter_5Entermap/TensorArray_9:1*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context*
T0*
is_constant( 
t
map/while_1/MergeMergemap/while_1/Entermap/while_1/NextIteration*
T0*
N*
_output_shapes
: : 
z
map/while_1/Merge_1Mergemap/while_1/Enter_1map/while_1/NextIteration_1*
N*
_output_shapes
: : *
T0
z
map/while_1/Merge_2Mergemap/while_1/Enter_2map/while_1/NextIteration_2*
T0*
N*
_output_shapes
: : 
z
map/while_1/Merge_3Mergemap/while_1/Enter_3map/while_1/NextIteration_3*
N*
_output_shapes
: : *
T0
z
map/while_1/Merge_4Mergemap/while_1/Enter_4map/while_1/NextIteration_4*
N*
_output_shapes
: : *
T0
z
map/while_1/Merge_5Mergemap/while_1/Enter_5map/while_1/NextIteration_5*
T0*
N*
_output_shapes
: : 
d
map/while_1/LessLessmap/while_1/Mergemap/while_1/Less/Enter*
_output_shapes
: *
T0
?
map/while_1/Less/EnterEntermap/strided_slice*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
h
map/while_1/Less_1Lessmap/while_1/Merge_1map/while_1/Less/Enter*
T0*
_output_shapes
: 
b
map/while_1/LogicalAnd
LogicalAndmap/while_1/Lessmap/while_1/Less_1*
_output_shapes
: 
P
map/while_1/LoopCondLoopCondmap/while_1/LogicalAnd*
_output_shapes
: 
?
map/while_1/SwitchSwitchmap/while_1/Mergemap/while_1/LoopCond*
T0*$
_class
loc:@map/while_1/Merge*
_output_shapes
: : 
?
map/while_1/Switch_1Switchmap/while_1/Merge_1map/while_1/LoopCond*
T0*&
_class
loc:@map/while_1/Merge_1*
_output_shapes
: : 
?
map/while_1/Switch_2Switchmap/while_1/Merge_2map/while_1/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map/while_1/Merge_2
?
map/while_1/Switch_3Switchmap/while_1/Merge_3map/while_1/LoopCond*
T0*&
_class
loc:@map/while_1/Merge_3*
_output_shapes
: : 
?
map/while_1/Switch_4Switchmap/while_1/Merge_4map/while_1/LoopCond*
_output_shapes
: : *
T0*&
_class
loc:@map/while_1/Merge_4
?
map/while_1/Switch_5Switchmap/while_1/Merge_5map/while_1/LoopCond*
T0*&
_class
loc:@map/while_1/Merge_5*
_output_shapes
: : 
W
map/while_1/IdentityIdentitymap/while_1/Switch:1*
T0*
_output_shapes
: 
[
map/while_1/Identity_1Identitymap/while_1/Switch_1:1*
_output_shapes
: *
T0
[
map/while_1/Identity_2Identitymap/while_1/Switch_2:1*
T0*
_output_shapes
: 
[
map/while_1/Identity_3Identitymap/while_1/Switch_3:1*
T0*
_output_shapes
: 
[
map/while_1/Identity_4Identitymap/while_1/Switch_4:1*
_output_shapes
: *
T0
[
map/while_1/Identity_5Identitymap/while_1/Switch_5:1*
T0*
_output_shapes
: 
j
map/while_1/add/yConst^map/while_1/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map/while_1/addAddmap/while_1/Identitymap/while_1/add/y*
T0*
_output_shapes
: 
?
map/while_1/TensorArrayReadV3TensorArrayReadV3#map/while_1/TensorArrayReadV3/Entermap/while_1/Identity_1%map/while_1/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
:
?
#map/while_1/TensorArrayReadV3/EnterEntermap/TensorArray_2*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*
is_constant(
?
%map/while_1/TensorArrayReadV3/Enter_1Entermap/while/Exit_2*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/TensorArrayReadV3_1TensorArrayReadV3%map/while_1/TensorArrayReadV3_1/Entermap/while_1/Identity_1'map/while_1/TensorArrayReadV3_1/Enter_1*
dtype0*
_output_shapes
:
?
%map/while_1/TensorArrayReadV3_1/EnterEntermap/TensorArray_3*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*
is_constant(
?
'map/while_1/TensorArrayReadV3_1/Enter_1Entermap/while/Exit_3*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/TensorArrayReadV3_2TensorArrayReadV3%map/while_1/TensorArrayReadV3_2/Entermap/while_1/Identity_1'map/while_1/TensorArrayReadV3_2/Enter_1*
dtype0*
_output_shapes
:
?
%map/while_1/TensorArrayReadV3_2/EnterEntermap/TensorArray_4*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context
?
'map/while_1/TensorArrayReadV3_2/Enter_1Entermap/while/Exit_4*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/TensorArrayReadV3_3TensorArrayReadV3%map/while_1/TensorArrayReadV3_3/Entermap/while_1/Identity_1'map/while_1/TensorArrayReadV3_3/Enter_1*
dtype0*
_output_shapes
:
?
%map/while_1/TensorArrayReadV3_3/EnterEntermap/TensorArray_5*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*
is_constant(
?
'map/while_1/TensorArrayReadV3_3/Enter_1Entermap/while/Exit_5*
T0*
is_constant(*
parallel_iterations*
_output_shapes
: *)

frame_namemap/while_1/while_context
?
map/while_1/unstackUnpackmap/while_1/unstack/Enter^map/while_1/Identity*
_output_shapes
: : : *	
num*
T0*

axis 
?
map/while_1/unstack/EnterEntermap/while/Exit_6*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context
w
map/while_1/ShapeShapemap/while_1/TensorArrayReadV3*
T0*
out_type0*#
_output_shapes
:?????????
?
map/while_1/strided_slice/stackConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
!map/while_1/strided_slice/stack_1Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
!map/while_1/strided_slice/stack_2Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while_1/strided_sliceStridedSlicemap/while_1/Shapemap/while_1/strided_slice/stack!map/while_1/strided_slice/stack_1!map/while_1/strided_slice/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
g
map/while_1/subSubmap/while_1/unstackmap/while_1/strided_slice*
T0*
_output_shapes
: 
?
!map/while_1/strided_slice_1/stackConst^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_1/stack_1Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
#map/while_1/strided_slice_1/stack_2Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while_1/strided_slice_1StridedSlicemap/while_1/Shape!map/while_1/strided_slice_1/stack#map/while_1/strided_slice_1/stack_1#map/while_1/strided_slice_1/stack_2*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
map/while_1/sub_1Submap/while_1/unstack:1map/while_1/strided_slice_1*
_output_shapes
: *
T0
?
!map/while_1/strided_slice_2/stackConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
#map/while_1/strided_slice_2/stack_1Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_2/stack_2Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while_1/strided_slice_2StridedSlicemap/while_1/Shape!map/while_1/strided_slice_2/stack#map/while_1/strided_slice_2/stack_1#map/while_1/strided_slice_2/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
map/while_1/sub_2Submap/while_1/unstack:2map/while_1/strided_slice_2*
T0*
_output_shapes
: 
?
!map/while_1/zeros/shape_as_tensorConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
p
map/while_1/zeros/ConstConst^map/while_1/Identity*
dtype0*
_output_shapes
: *
value	B : 
?
map/while_1/zerosFill!map/while_1/zeros/shape_as_tensormap/while_1/zeros/Const*
T0*

index_type0*
_output_shapes
:
?
map/while_1/stack/values_1Packmap/while_1/submap/while_1/sub_1map/while_1/sub_2*
N*
_output_shapes
:*
T0*

axis 
?
map/while_1/stackPackmap/while_1/zerosmap/while_1/stack/values_1*
N*
_output_shapes

:*
T0*

axis
?
map/while_1/PadPadmap/while_1/TensorArrayReadV3map/while_1/stack*
	Tpaddings0*=
_output_shapes+
):'???????????????????????????*
T0
?
/map/while_1/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35map/while_1/TensorArrayWrite/TensorArrayWriteV3/Entermap/while_1/Identity_1map/while_1/Padmap/while_1/Identity_2*
_output_shapes
: *
T0*"
_class
loc:@map/while_1/Pad
?
5map/while_1/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_6*)

frame_namemap/while_1/while_context*
_output_shapes
:*
T0*"
_class
loc:@map/while_1/Pad*
parallel_iterations*
is_constant(
?
map/while_1/unstack_1Unpackmap/while_1/unstack_1/Enter^map/while_1/Identity*	
num*
T0*

axis *
_output_shapes
: : : 
?
map/while_1/unstack_1/EnterEntermap/while/Exit_7*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context
{
map/while_1/Shape_1Shapemap/while_1/TensorArrayReadV3_1*
T0*
out_type0*#
_output_shapes
:?????????
?
!map/while_1/strided_slice_3/stackConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
#map/while_1/strided_slice_3/stack_1Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
#map/while_1/strided_slice_3/stack_2Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while_1/strided_slice_3StridedSlicemap/while_1/Shape_1!map/while_1/strided_slice_3/stack#map/while_1/strided_slice_3/stack_1#map/while_1/strided_slice_3/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
m
map/while_1/sub_3Submap/while_1/unstack_1map/while_1/strided_slice_3*
T0*
_output_shapes
: 
?
!map/while_1/strided_slice_4/stackConst^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_4/stack_1Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
#map/while_1/strided_slice_4/stack_2Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while_1/strided_slice_4StridedSlicemap/while_1/Shape_1!map/while_1/strided_slice_4/stack#map/while_1/strided_slice_4/stack_1#map/while_1/strided_slice_4/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
o
map/while_1/sub_4Submap/while_1/unstack_1:1map/while_1/strided_slice_4*
_output_shapes
: *
T0
?
!map/while_1/strided_slice_5/stackConst^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_5/stack_1Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_5/stack_2Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
map/while_1/strided_slice_5StridedSlicemap/while_1/Shape_1!map/while_1/strided_slice_5/stack#map/while_1/strided_slice_5/stack_1#map/while_1/strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
o
map/while_1/sub_5Submap/while_1/unstack_1:2map/while_1/strided_slice_5*
_output_shapes
: *
T0
?
#map/while_1/zeros_1/shape_as_tensorConst^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
r
map/while_1/zeros_1/ConstConst^map/while_1/Identity*
value	B : *
dtype0*
_output_shapes
: 
?
map/while_1/zeros_1Fill#map/while_1/zeros_1/shape_as_tensormap/while_1/zeros_1/Const*
T0*

index_type0*
_output_shapes
:
?
map/while_1/stack_1/values_1Packmap/while_1/sub_3map/while_1/sub_4map/while_1/sub_5*
T0*

axis *
N*
_output_shapes
:
?
map/while_1/stack_1Packmap/while_1/zeros_1map/while_1/stack_1/values_1*
N*
_output_shapes

:*
T0*

axis
?
map/while_1/Pad_1Padmap/while_1/TensorArrayReadV3_1map/while_1/stack_1*
T0*
	Tpaddings0*=
_output_shapes+
):'???????????????????????????
?
1map/while_1/TensorArrayWrite_1/TensorArrayWriteV3TensorArrayWriteV37map/while_1/TensorArrayWrite_1/TensorArrayWriteV3/Entermap/while_1/Identity_1map/while_1/Pad_1map/while_1/Identity_3*
T0*$
_class
loc:@map/while_1/Pad_1*
_output_shapes
: 
?
7map/while_1/TensorArrayWrite_1/TensorArrayWriteV3/EnterEntermap/TensorArray_7*
is_constant(*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*$
_class
loc:@map/while_1/Pad_1*
parallel_iterations
?
map/while_1/unstack_2Unpackmap/while_1/unstack_2/Enter^map/while_1/Identity*
_output_shapes
: *	
num*
T0*

axis 
?
map/while_1/unstack_2/EnterEntermap/while/Exit_8*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*
is_constant(
{
map/while_1/Shape_2Shapemap/while_1/TensorArrayReadV3_2*#
_output_shapes
:?????????*
T0*
out_type0
?
!map/while_1/strided_slice_6/stackConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB: 
?
#map/while_1/strided_slice_6/stack_1Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_6/stack_2Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while_1/strided_slice_6StridedSlicemap/while_1/Shape_2!map/while_1/strided_slice_6/stack#map/while_1/strided_slice_6/stack_1#map/while_1/strided_slice_6/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
Index0*
T0
m
map/while_1/sub_6Submap/while_1/unstack_2map/while_1/strided_slice_6*
_output_shapes
: *
T0
?
#map/while_1/zeros_2/shape_as_tensorConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
r
map/while_1/zeros_2/ConstConst^map/while_1/Identity*
dtype0*
_output_shapes
: *
value	B : 
?
map/while_1/zeros_2Fill#map/while_1/zeros_2/shape_as_tensormap/while_1/zeros_2/Const*
T0*

index_type0*
_output_shapes
:
q
map/while_1/stack_2/values_1Packmap/while_1/sub_6*
T0*

axis *
N*
_output_shapes
:
?
map/while_1/stack_2Packmap/while_1/zeros_2map/while_1/stack_2/values_1*
N*
_output_shapes

:*
T0*

axis
?
map/while_1/Pad_2Padmap/while_1/TensorArrayReadV3_2map/while_1/stack_2*
	Tpaddings0*#
_output_shapes
:?????????*
T0
?
1map/while_1/TensorArrayWrite_2/TensorArrayWriteV3TensorArrayWriteV37map/while_1/TensorArrayWrite_2/TensorArrayWriteV3/Entermap/while_1/Identity_1map/while_1/Pad_2map/while_1/Identity_4*
_output_shapes
: *
T0*$
_class
loc:@map/while_1/Pad_2
?
7map/while_1/TensorArrayWrite_2/TensorArrayWriteV3/EnterEntermap/TensorArray_8*
parallel_iterations*
is_constant(*
_output_shapes
:*)

frame_namemap/while_1/while_context*
T0*$
_class
loc:@map/while_1/Pad_2
?
map/while_1/unstack_3Unpackmap/while_1/unstack_3/Enter^map/while_1/Identity*
_output_shapes
: *	
num*
T0*

axis 
?
map/while_1/unstack_3/EnterEntermap/while/Exit_9*
T0*
is_constant(*
parallel_iterations*
_output_shapes
:*)

frame_namemap/while_1/while_context
{
map/while_1/Shape_3Shapemap/while_1/TensorArrayReadV3_3*#
_output_shapes
:?????????*
T0*
out_type0
?
!map/while_1/strided_slice_7/stackConst^map/while_1/Identity*
valueB: *
dtype0*
_output_shapes
:
?
#map/while_1/strided_slice_7/stack_1Const^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
?
#map/while_1/strided_slice_7/stack_2Const^map/while_1/Identity*
valueB:*
dtype0*
_output_shapes
:
?
map/while_1/strided_slice_7StridedSlicemap/while_1/Shape_3!map/while_1/strided_slice_7/stack#map/while_1/strided_slice_7/stack_1#map/while_1/strided_slice_7/stack_2*
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
m
map/while_1/sub_7Submap/while_1/unstack_3map/while_1/strided_slice_7*
T0*
_output_shapes
: 
?
#map/while_1/zeros_3/shape_as_tensorConst^map/while_1/Identity*
dtype0*
_output_shapes
:*
valueB:
r
map/while_1/zeros_3/ConstConst^map/while_1/Identity*
value	B : *
dtype0*
_output_shapes
: 
?
map/while_1/zeros_3Fill#map/while_1/zeros_3/shape_as_tensormap/while_1/zeros_3/Const*
_output_shapes
:*
T0*

index_type0
q
map/while_1/stack_3/values_1Packmap/while_1/sub_7*
T0*

axis *
N*
_output_shapes
:
?
map/while_1/stack_3Packmap/while_1/zeros_3map/while_1/stack_3/values_1*
T0*

axis*
N*
_output_shapes

:
?
map/while_1/Pad_3Padmap/while_1/TensorArrayReadV3_3map/while_1/stack_3*
	Tpaddings0*#
_output_shapes
:?????????*
T0
?
1map/while_1/TensorArrayWrite_3/TensorArrayWriteV3TensorArrayWriteV37map/while_1/TensorArrayWrite_3/TensorArrayWriteV3/Entermap/while_1/Identity_1map/while_1/Pad_3map/while_1/Identity_5*
T0*$
_class
loc:@map/while_1/Pad_3*
_output_shapes
: 
?
7map/while_1/TensorArrayWrite_3/TensorArrayWriteV3/EnterEntermap/TensorArray_9*
is_constant(*)

frame_namemap/while_1/while_context*
_output_shapes
:*
T0*$
_class
loc:@map/while_1/Pad_3*
parallel_iterations
l
map/while_1/add_1/yConst^map/while_1/Identity*
dtype0*
_output_shapes
: *
value	B :
f
map/while_1/add_1Addmap/while_1/Identity_1map/while_1/add_1/y*
T0*
_output_shapes
: 
\
map/while_1/NextIterationNextIterationmap/while_1/add*
T0*
_output_shapes
: 
`
map/while_1/NextIteration_1NextIterationmap/while_1/add_1*
_output_shapes
: *
T0
~
map/while_1/NextIteration_2NextIteration/map/while_1/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *
T0
?
map/while_1/NextIteration_3NextIteration1map/while_1/TensorArrayWrite_1/TensorArrayWriteV3*
T0*
_output_shapes
: 
?
map/while_1/NextIteration_4NextIteration1map/while_1/TensorArrayWrite_2/TensorArrayWriteV3*
T0*
_output_shapes
: 
?
map/while_1/NextIteration_5NextIteration1map/while_1/TensorArrayWrite_3/TensorArrayWriteV3*
T0*
_output_shapes
: 
M
map/while_1/ExitExitmap/while_1/Switch*
_output_shapes
: *
T0
Q
map/while_1/Exit_1Exitmap/while_1/Switch_1*
T0*
_output_shapes
: 
Q
map/while_1/Exit_2Exitmap/while_1/Switch_2*
_output_shapes
: *
T0
Q
map/while_1/Exit_3Exitmap/while_1/Switch_3*
T0*
_output_shapes
: 
Q
map/while_1/Exit_4Exitmap/while_1/Switch_4*
_output_shapes
: *
T0
Q
map/while_1/Exit_5Exitmap/while_1/Switch_5*
T0*
_output_shapes
: 
?
&map/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map/TensorArray_6map/while_1/Exit_2*$
_class
loc:@map/TensorArray_6*
_output_shapes
: 
?
 map/TensorArrayStack/range/startConst*$
_class
loc:@map/TensorArray_6*
value	B : *
dtype0*
_output_shapes
: 
?
 map/TensorArrayStack/range/deltaConst*$
_class
loc:@map/TensorArray_6*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayStack/rangeRange map/TensorArrayStack/range/start&map/TensorArrayStack/TensorArraySizeV3 map/TensorArrayStack/range/delta*$
_class
loc:@map/TensorArray_6*#
_output_shapes
:?????????*

Tidx0
?
(map/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_6map/TensorArrayStack/rangemap/while_1/Exit_2*
dtype0*1
_output_shapes
:???????????*:
element_shape):'???????????????????????????*$
_class
loc:@map/TensorArray_6
?
(map/TensorArrayStack_1/TensorArraySizeV3TensorArraySizeV3map/TensorArray_7map/while_1/Exit_3*$
_class
loc:@map/TensorArray_7*
_output_shapes
: 
?
"map/TensorArrayStack_1/range/startConst*$
_class
loc:@map/TensorArray_7*
value	B : *
dtype0*
_output_shapes
: 
?
"map/TensorArrayStack_1/range/deltaConst*$
_class
loc:@map/TensorArray_7*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayStack_1/rangeRange"map/TensorArrayStack_1/range/start(map/TensorArrayStack_1/TensorArraySizeV3"map/TensorArrayStack_1/range/delta*$
_class
loc:@map/TensorArray_7*#
_output_shapes
:?????????*

Tidx0
?
*map/TensorArrayStack_1/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_7map/TensorArrayStack_1/rangemap/while_1/Exit_3*
dtype0*A
_output_shapes/
-:+???????????????????????????*:
element_shape):'???????????????????????????*$
_class
loc:@map/TensorArray_7
?
(map/TensorArrayStack_2/TensorArraySizeV3TensorArraySizeV3map/TensorArray_8map/while_1/Exit_4*$
_class
loc:@map/TensorArray_8*
_output_shapes
: 
?
"map/TensorArrayStack_2/range/startConst*
dtype0*
_output_shapes
: *$
_class
loc:@map/TensorArray_8*
value	B : 
?
"map/TensorArrayStack_2/range/deltaConst*$
_class
loc:@map/TensorArray_8*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayStack_2/rangeRange"map/TensorArrayStack_2/range/start(map/TensorArrayStack_2/TensorArraySizeV3"map/TensorArrayStack_2/range/delta*#
_output_shapes
:?????????*

Tidx0*$
_class
loc:@map/TensorArray_8
?
*map/TensorArrayStack_2/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_8map/TensorArrayStack_2/rangemap/while_1/Exit_4*
dtype0*0
_output_shapes
:??????????????????* 
element_shape:?????????*$
_class
loc:@map/TensorArray_8
?
(map/TensorArrayStack_3/TensorArraySizeV3TensorArraySizeV3map/TensorArray_9map/while_1/Exit_5*
_output_shapes
: *$
_class
loc:@map/TensorArray_9
?
"map/TensorArrayStack_3/range/startConst*
dtype0*
_output_shapes
: *$
_class
loc:@map/TensorArray_9*
value	B : 
?
"map/TensorArrayStack_3/range/deltaConst*$
_class
loc:@map/TensorArray_9*
value	B :*
dtype0*
_output_shapes
: 
?
map/TensorArrayStack_3/rangeRange"map/TensorArrayStack_3/range/start(map/TensorArrayStack_3/TensorArraySizeV3"map/TensorArrayStack_3/range/delta*$
_class
loc:@map/TensorArray_9*#
_output_shapes
:?????????*

Tidx0
?
*map/TensorArrayStack_3/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_9map/TensorArrayStack_3/rangemap/while_1/Exit_5* 
element_shape:?????????*$
_class
loc:@map/TensorArray_9*
dtype0*0
_output_shapes
:??????????????????
?

map/Cast_1Cast*map/TensorArrayStack_2/TensorArrayGatherV3*
Truncate( *'
_output_shapes
:?????????*

DstT0*

SrcT0
?

map/Cast_2Cast*map/TensorArrayStack_3/TensorArrayGatherV3*
Truncate( *'
_output_shapes
:?????????*

DstT0*

SrcT0
n
,resnet_v1_50/ImageMask/Max/reduction_indicesConst*
dtype0*
_output_shapes
: *
value	B : 
?
resnet_v1_50/ImageMask/MaxMax
map/Cast_2,resnet_v1_50/ImageMask/Max/reduction_indices*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
t
*resnet_v1_50/ImageMask/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
v
,resnet_v1_50/ImageMask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,resnet_v1_50/ImageMask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
$resnet_v1_50/ImageMask/strided_sliceStridedSliceresnet_v1_50/ImageMask/Max*resnet_v1_50/ImageMask/strided_slice/stack,resnet_v1_50/ImageMask/strided_slice/stack_1,resnet_v1_50/ImageMask/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
"resnet_v1_50/ImageMask/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"resnet_v1_50/ImageMask/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
resnet_v1_50/ImageMask/rangeRange"resnet_v1_50/ImageMask/range/start$resnet_v1_50/ImageMask/strided_slice"resnet_v1_50/ImageMask/range/delta*#
_output_shapes
:?????????*

Tidx0
v
,resnet_v1_50/ImageMask/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
x
.resnet_v1_50/ImageMask/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
x
.resnet_v1_50/ImageMask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
&resnet_v1_50/ImageMask/strided_slice_1StridedSliceresnet_v1_50/ImageMask/Max,resnet_v1_50/ImageMask/strided_slice_1/stack.resnet_v1_50/ImageMask/strided_slice_1/stack_1.resnet_v1_50/ImageMask/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
f
$resnet_v1_50/ImageMask/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$resnet_v1_50/ImageMask/range_1/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
resnet_v1_50/ImageMask/range_1Range$resnet_v1_50/ImageMask/range_1/start&resnet_v1_50/ImageMask/strided_slice_1$resnet_v1_50/ImageMask/range_1/delta*#
_output_shapes
:?????????*

Tidx0
~
-resnet_v1_50/ImageMask/meshgrid/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
'resnet_v1_50/ImageMask/meshgrid/ReshapeReshaperesnet_v1_50/ImageMask/range-resnet_v1_50/ImageMask/meshgrid/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
/resnet_v1_50/ImageMask/meshgrid/Reshape_1/shapeConst*
valueB"   ????*
dtype0*
_output_shapes
:
?
)resnet_v1_50/ImageMask/meshgrid/Reshape_1Reshaperesnet_v1_50/ImageMask/range_1/resnet_v1_50/ImageMask/meshgrid/Reshape_1/shape*'
_output_shapes
:?????????*
T0*
Tshape0
{
$resnet_v1_50/ImageMask/meshgrid/SizeSizeresnet_v1_50/ImageMask/range*
_output_shapes
: *
T0*
out_type0

&resnet_v1_50/ImageMask/meshgrid/Size_1Sizeresnet_v1_50/ImageMask/range_1*
T0*
out_type0*
_output_shapes
: 
?
/resnet_v1_50/ImageMask/meshgrid/Reshape_2/shapeConst*
valueB"   ????*
dtype0*
_output_shapes
:
?
)resnet_v1_50/ImageMask/meshgrid/Reshape_2Reshape'resnet_v1_50/ImageMask/meshgrid/Reshape/resnet_v1_50/ImageMask/meshgrid/Reshape_2/shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
/resnet_v1_50/ImageMask/meshgrid/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
)resnet_v1_50/ImageMask/meshgrid/Reshape_3Reshape)resnet_v1_50/ImageMask/meshgrid/Reshape_1/resnet_v1_50/ImageMask/meshgrid/Reshape_3/shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
(resnet_v1_50/ImageMask/meshgrid/ones/mulMul&resnet_v1_50/ImageMask/meshgrid/Size_1$resnet_v1_50/ImageMask/meshgrid/Size*
_output_shapes
: *
T0
n
+resnet_v1_50/ImageMask/meshgrid/ones/Less/yConst*
value
B :?*
dtype0*
_output_shapes
: 
?
)resnet_v1_50/ImageMask/meshgrid/ones/LessLess(resnet_v1_50/ImageMask/meshgrid/ones/mul+resnet_v1_50/ImageMask/meshgrid/ones/Less/y*
_output_shapes
: *
T0
?
+resnet_v1_50/ImageMask/meshgrid/ones/packedPack&resnet_v1_50/ImageMask/meshgrid/Size_1$resnet_v1_50/ImageMask/meshgrid/Size*
T0*

axis *
N*
_output_shapes
:
l
*resnet_v1_50/ImageMask/meshgrid/ones/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
?
$resnet_v1_50/ImageMask/meshgrid/onesFill+resnet_v1_50/ImageMask/meshgrid/ones/packed*resnet_v1_50/ImageMask/meshgrid/ones/Const*0
_output_shapes
:??????????????????*
T0*

index_type0
?
#resnet_v1_50/ImageMask/meshgrid/mulMul)resnet_v1_50/ImageMask/meshgrid/Reshape_2$resnet_v1_50/ImageMask/meshgrid/ones*0
_output_shapes
:??????????????????*
T0
?
%resnet_v1_50/ImageMask/meshgrid/mul_1Mul)resnet_v1_50/ImageMask/meshgrid/Reshape_3$resnet_v1_50/ImageMask/meshgrid/ones*0
_output_shapes
:??????????????????*
T0
}
,resnet_v1_50/ImageMask/strided_slice_2/stackConst*
valueB"       *
dtype0*
_output_shapes
:

.resnet_v1_50/ImageMask/strided_slice_2/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:

.resnet_v1_50/ImageMask/strided_slice_2/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
&resnet_v1_50/ImageMask/strided_slice_2StridedSlice
map/Cast_2,resnet_v1_50/ImageMask/strided_slice_2/stack.resnet_v1_50/ImageMask/strided_slice_2/stack_1.resnet_v1_50/ImageMask/strided_slice_2/stack_2*
end_mask*#
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
?
,resnet_v1_50/ImageMask/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*!
valueB"            
?
.resnet_v1_50/ImageMask/strided_slice_3/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
?
&resnet_v1_50/ImageMask/strided_slice_3StridedSlice#resnet_v1_50/ImageMask/meshgrid/mul,resnet_v1_50/ImageMask/strided_slice_3/stack.resnet_v1_50/ImageMask/strided_slice_3/stack_1.resnet_v1_50/ImageMask/strided_slice_3/stack_2*
end_mask*4
_output_shapes"
 :??????????????????*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask
?
,resnet_v1_50/ImageMask/strided_slice_4/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_4/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"            
?
.resnet_v1_50/ImageMask/strided_slice_4/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
?
&resnet_v1_50/ImageMask/strided_slice_4StridedSlice&resnet_v1_50/ImageMask/strided_slice_2,resnet_v1_50/ImageMask/strided_slice_4/stack.resnet_v1_50/ImageMask/strided_slice_4/stack_1.resnet_v1_50/ImageMask/strided_slice_4/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask*
end_mask*+
_output_shapes
:?????????
?
resnet_v1_50/ImageMask/LessLess&resnet_v1_50/ImageMask/strided_slice_3&resnet_v1_50/ImageMask/strided_slice_4*
T0*=
_output_shapes+
):'???????????????????????????
}
,resnet_v1_50/ImageMask/strided_slice_5/stackConst*
dtype0*
_output_shapes
:*
valueB"        

.resnet_v1_50/ImageMask/strided_slice_5/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:

.resnet_v1_50/ImageMask/strided_slice_5/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
&resnet_v1_50/ImageMask/strided_slice_5StridedSlice
map/Cast_2,resnet_v1_50/ImageMask/strided_slice_5/stack.resnet_v1_50/ImageMask/strided_slice_5/stack_1.resnet_v1_50/ImageMask/strided_slice_5/stack_2*
end_mask*#
_output_shapes
:?????????*
Index0*
T0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask 
?
,resnet_v1_50/ImageMask/strided_slice_6/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_6/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_6/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
?
&resnet_v1_50/ImageMask/strided_slice_6StridedSlice%resnet_v1_50/ImageMask/meshgrid/mul_1,resnet_v1_50/ImageMask/strided_slice_6/stack.resnet_v1_50/ImageMask/strided_slice_6/stack_1.resnet_v1_50/ImageMask/strided_slice_6/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask*
end_mask*4
_output_shapes"
 :??????????????????*
T0*
Index0
?
,resnet_v1_50/ImageMask/strided_slice_7/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_7/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
?
.resnet_v1_50/ImageMask/strided_slice_7/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
?
&resnet_v1_50/ImageMask/strided_slice_7StridedSlice&resnet_v1_50/ImageMask/strided_slice_5,resnet_v1_50/ImageMask/strided_slice_7/stack.resnet_v1_50/ImageMask/strided_slice_7/stack_1.resnet_v1_50/ImageMask/strided_slice_7/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask*
end_mask*+
_output_shapes
:?????????
?
resnet_v1_50/ImageMask/Less_1Less&resnet_v1_50/ImageMask/strided_slice_6&resnet_v1_50/ImageMask/strided_slice_7*
T0*=
_output_shapes+
):'???????????????????????????
?
!resnet_v1_50/ImageMask/LogicalAnd
LogicalAndresnet_v1_50/ImageMask/Lessresnet_v1_50/ImageMask/Less_1*=
_output_shapes+
):'???????????????????????????
?
resnet_v1_50/ImageMask/CastCast!resnet_v1_50/ImageMask/LogicalAnd*

SrcT0
*
Truncate( *=
_output_shapes+
):'???????????????????????????*

DstT0
y
 resnet_v1_50/strided_slice/stackConst*%
valueB"                *
dtype0*
_output_shapes
:
{
"resnet_v1_50/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*%
valueB"                
{
"resnet_v1_50/strided_slice/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
?
resnet_v1_50/strided_sliceStridedSliceresnet_v1_50/ImageMask/Cast resnet_v1_50/strided_slice/stack"resnet_v1_50/strided_slice/stack_1"resnet_v1_50/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask*
end_mask*A
_output_shapes/
-:+???????????????????????????*
T0*
Index0
?
resnet_v1_50/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             
?
resnet_v1_50/PadPad(map/TensorArrayStack/TensorArrayGatherV3resnet_v1_50/Pad/paddings*
T0*
	Tpaddings0*1
_output_shapes
:???????????
?
=resnet_v1_50/conv1/weights/Initializer/truncated_normal/shapeConst*-
_class#
!loc:@resnet_v1_50/conv1/weights*%
valueB"         @   *
dtype0*
_output_shapes
:
?
<resnet_v1_50/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@resnet_v1_50/conv1/weights*
valueB
 *    
?
>resnet_v1_50/conv1/weights/Initializer/truncated_normal/stddevConst*-
_class#
!loc:@resnet_v1_50/conv1/weights*
valueB
 *A/>*
dtype0*
_output_shapes
: 
?
Gresnet_v1_50/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal=resnet_v1_50/conv1/weights/Initializer/truncated_normal/shape*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights*
seed2 *
dtype0*&
_output_shapes
:@*

seed 
?
;resnet_v1_50/conv1/weights/Initializer/truncated_normal/mulMulGresnet_v1_50/conv1/weights/Initializer/truncated_normal/TruncatedNormal>resnet_v1_50/conv1/weights/Initializer/truncated_normal/stddev*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights*&
_output_shapes
:@
?
7resnet_v1_50/conv1/weights/Initializer/truncated_normalAdd;resnet_v1_50/conv1/weights/Initializer/truncated_normal/mul<resnet_v1_50/conv1/weights/Initializer/truncated_normal/mean*&
_output_shapes
:@*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights
?
resnet_v1_50/conv1/weights
VariableV2*
	container *
shape:@*
dtype0*&
_output_shapes
:@*
shared_name *-
_class#
!loc:@resnet_v1_50/conv1/weights
?
!resnet_v1_50/conv1/weights/AssignAssignresnet_v1_50/conv1/weights7resnet_v1_50/conv1/weights/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights
?
resnet_v1_50/conv1/weights/readIdentityresnet_v1_50/conv1/weights*&
_output_shapes
:@*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights
?
:resnet_v1_50/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@resnet_v1_50/conv1/weights*
valueB
 *??8
?
;resnet_v1_50/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Lossresnet_v1_50/conv1/weights/read*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights*
_output_shapes
: 
?
4resnet_v1_50/conv1/kernel/Regularizer/l2_regularizerMul:resnet_v1_50/conv1/kernel/Regularizer/l2_regularizer/scale;resnet_v1_50/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights*
_output_shapes
: 
q
 resnet_v1_50/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
resnet_v1_50/conv1/Conv2DConv2Dresnet_v1_50/Padresnet_v1_50/conv1/weights/read*/
_output_shapes
:?????????pp@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
?
3resnet_v1_50/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*5
_class+
)'loc:@resnet_v1_50/conv1/BatchNorm/gamma*
valueB@*  ??
?
"resnet_v1_50/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *5
_class+
)'loc:@resnet_v1_50/conv1/BatchNorm/gamma*
	container *
shape:@
?
)resnet_v1_50/conv1/BatchNorm/gamma/AssignAssign"resnet_v1_50/conv1/BatchNorm/gamma3resnet_v1_50/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*5
_class+
)'loc:@resnet_v1_50/conv1/BatchNorm/gamma
?
'resnet_v1_50/conv1/BatchNorm/gamma/readIdentity"resnet_v1_50/conv1/BatchNorm/gamma*
_output_shapes
:@*
T0*5
_class+
)'loc:@resnet_v1_50/conv1/BatchNorm/gamma
?
3resnet_v1_50/conv1/BatchNorm/beta/Initializer/zerosConst*4
_class*
(&loc:@resnet_v1_50/conv1/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
?
!resnet_v1_50/conv1/BatchNorm/beta
VariableV2*
shared_name *4
_class*
(&loc:@resnet_v1_50/conv1/BatchNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
(resnet_v1_50/conv1/BatchNorm/beta/AssignAssign!resnet_v1_50/conv1/BatchNorm/beta3resnet_v1_50/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@resnet_v1_50/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
&resnet_v1_50/conv1/BatchNorm/beta/readIdentity!resnet_v1_50/conv1/BatchNorm/beta*
T0*4
_class*
(&loc:@resnet_v1_50/conv1/BatchNorm/beta*
_output_shapes
:@
?
:resnet_v1_50/conv1/BatchNorm/moving_mean/Initializer/zerosConst*;
_class1
/-loc:@resnet_v1_50/conv1/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
?
(resnet_v1_50/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *;
_class1
/-loc:@resnet_v1_50/conv1/BatchNorm/moving_mean*
	container *
shape:@
?
/resnet_v1_50/conv1/BatchNorm/moving_mean/AssignAssign(resnet_v1_50/conv1/BatchNorm/moving_mean:resnet_v1_50/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*;
_class1
/-loc:@resnet_v1_50/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
-resnet_v1_50/conv1/BatchNorm/moving_mean/readIdentity(resnet_v1_50/conv1/BatchNorm/moving_mean*
T0*;
_class1
/-loc:@resnet_v1_50/conv1/BatchNorm/moving_mean*
_output_shapes
:@
?
=resnet_v1_50/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*?
_class5
31loc:@resnet_v1_50/conv1/BatchNorm/moving_variance*
valueB@*  ??
?
,resnet_v1_50/conv1/BatchNorm/moving_variance
VariableV2*
shared_name *?
_class5
31loc:@resnet_v1_50/conv1/BatchNorm/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
3resnet_v1_50/conv1/BatchNorm/moving_variance/AssignAssign,resnet_v1_50/conv1/BatchNorm/moving_variance=resnet_v1_50/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*?
_class5
31loc:@resnet_v1_50/conv1/BatchNorm/moving_variance
?
1resnet_v1_50/conv1/BatchNorm/moving_variance/readIdentity,resnet_v1_50/conv1/BatchNorm/moving_variance*
_output_shapes
:@*
T0*?
_class5
31loc:@resnet_v1_50/conv1/BatchNorm/moving_variance
?
+resnet_v1_50/conv1/BatchNorm/FusedBatchNormFusedBatchNormresnet_v1_50/conv1/Conv2D'resnet_v1_50/conv1/BatchNorm/gamma/read&resnet_v1_50/conv1/BatchNorm/beta/read-resnet_v1_50/conv1/BatchNorm/moving_mean/read1resnet_v1_50/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*G
_output_shapes5
3:?????????pp@:@:@:@:@*
is_training( 
g
"resnet_v1_50/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
resnet_v1_50/conv1/ReluRelu+resnet_v1_50/conv1/BatchNorm/FusedBatchNorm*
T0*/
_output_shapes
:?????????pp@
?
resnet_v1_50/pool1/MaxPoolMaxPoolresnet_v1_50/conv1/Relu*/
_output_shapes
:?????????88@*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingSAME
{
"resnet_v1_50/strided_slice_1/stackConst*%
valueB"                *
dtype0*
_output_shapes
:
}
$resnet_v1_50/strided_slice_1/stack_1Const*%
valueB"                *
dtype0*
_output_shapes
:
}
$resnet_v1_50/strided_slice_1/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
?
resnet_v1_50/strided_slice_1StridedSliceresnet_v1_50/strided_slice"resnet_v1_50/strided_slice_1/stack$resnet_v1_50/strided_slice_1/stack_1$resnet_v1_50/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*A
_output_shapes/
-:+???????????????????????????*
T0*
Index0
?
\resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shapeConst*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*%
valueB"      @      *
dtype0*
_output_shapes
:
?
[resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *    
?
]resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *?dN>
?
fresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal\resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@?*

seed *
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
seed2 
?
Zresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mulMulfresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormal]resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddev*
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*'
_output_shapes
:@?
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normalAddZresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mul[resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mean*
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*'
_output_shapes
:@?
?
9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights
VariableV2*
dtype0*'
_output_shapes
:@?*
shared_name *L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
	container *
shape:@?
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/AssignAssign9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weightsVresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal*
validate_shape(*'
_output_shapes
:@?*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/readIdentity9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*'
_output_shapes
:@?
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *??8
?
Zresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2LossL2Loss>resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/read*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights
?
Sresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizerMulYresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleZresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/Conv2DConv2Dresnet_v1_50/pool1/MaxPool>resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/read*
paddingSAME*0
_output_shapes
:?????????88?*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Rresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/onesConst*T
_classJ
HFloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Aresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *T
_classJ
HFloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
	container *
shape:?
?
Hresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignAssignAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaRresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readIdentityAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
_output_shapes	
:?*
T0*T
_classJ
HFloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
Rresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*S
_classI
GEloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB?*    
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *S
_classI
GEloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
Gresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignAssign@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/betaRresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Eresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readIdentity@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
_output_shapes	
:?*
T0*S
_classI
GEloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zerosConst*Z
_classP
NLloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Gresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Z
_classP
NLloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
Nresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignAssignGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanYresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
Lresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readIdentityGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*Z
_classP
NLloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
\resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/onesConst*^
_classT
RPloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *^
_classT
RPloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
?
Rresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignAssignKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance\resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Presnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/readIdentityKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
T0*^
_classT
RPloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
_output_shapes	
:?
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormFusedBatchNorm8resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/Conv2DFresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readEresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readLresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readPresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:?????????88?:?:?:?:?*
is_training( 
?
Aresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*%
valueB"      @   @   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
valueB
 *?dN>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*&
_output_shapes
:@@
?
Sresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*&
_output_shapes
:@@*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
VariableV2*
dtype0*&
_output_shapes
:@@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
	container *
shape:@@
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weightsSresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*&
_output_shapes
:@@
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
?
<resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2DConv2Dresnet_v1_50/pool1/MaxPool;resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????88@*
	dilations
*
T0
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
Cresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes
:@
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
valueB@*    
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:@
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
Bresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB@*    
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:@
?
Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes
:@
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB@*  ??
?
Hresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:@
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes
:@*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*/
_output_shapes
:?????????88@*
T0
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*%
valueB"      @   @   
?
Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
seed2 *
dtype0*&
_output_shapes
:@@*

seed 
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*&
_output_shapes
:@@*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights
?
Sresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*&
_output_shapes
:@@*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights
VariableV2*
dtype0*&
_output_shapes
:@@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
	container *
shape:@@
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weightsSresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
validate_shape(*&
_output_shapes
:@@
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Relu;resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/read*
paddingSAME*/
_output_shapes
:?????????88@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
valueB@*  ??
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
Cresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes
:@*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
valueB@*    
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB@*    
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
Iresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes
:@*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
Hresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:@
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
Mresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes
:@*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*/
_output_shapes
:?????????88@
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*%
valueB"      @      *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
valueB
 *?dN>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@?*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights
VariableV2*
dtype0*'
_output_shapes
:@?*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
	container *
shape:@?
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weightsSresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*'
_output_shapes
:@?*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*'
_output_shapes
:@?
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Relu;resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/read*0
_output_shapes
:?????????88?*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Cresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Iresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:?????????88?:?:?:?:?*
is_training( 
?
>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block1/unit_1/bottleneck_v1/addAddJresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormGresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*0
_output_shapes
:?????????88?*
T0
?
-resnet_v1_50/block1/unit_1/bottleneck_v1/ReluRelu,resnet_v1_50/block1/unit_1/bottleneck_v1/add*0
_output_shapes
:?????????88?*
T0
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*%
valueB"         @   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
valueB
 *?d?=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:?@*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*'
_output_shapes
:?@*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*'
_output_shapes
:?@
?
6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights
VariableV2*
dtype0*'
_output_shapes
:?@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
	container *
shape:?@
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weightsSresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*'
_output_shapes
:?@
?
;resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*'
_output_shapes
:?@
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block1/unit_1/bottleneck_v1/Relu;resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/read*
paddingSAME*/
_output_shapes
:?????????88@*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
Cresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes
:@
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
valueB@*    *
dtype0*
_output_shapes
:@
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
Bresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB@*    
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:@
?
Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
Iresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes
:@*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
Hresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes
:@*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*/
_output_shapes
:?????????88@
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*%
valueB"      @   @   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
Sresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights
VariableV2*
dtype0*&
_output_shapes
:@@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
	container *
shape:@@
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weightsSresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
validate_shape(*&
_output_shapes
:@@
?
;resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Relu;resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????88@*
	dilations
*
T0
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes
:@*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
valueB@*    
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:@
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes
:@*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB@*    
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes
:@
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes
:@*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB@*  ??
?
Hresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes
:@
?
Gresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*/
_output_shapes
:?????????88@*
T0
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*%
valueB"      @      
?
Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
valueB
 *?dN>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*'
_output_shapes
:@?*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights
VariableV2*
dtype0*'
_output_shapes
:@?*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
	container *
shape:@?
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weightsSresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
validate_shape(*'
_output_shapes
:@?
?
;resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Relu;resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/read*0
_output_shapes
:?????????88?*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
Cresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Vresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:?????????88?:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block1/unit_2/bottleneck_v1/addAdd-resnet_v1_50/block1/unit_1/bottleneck_v1/ReluGresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:?????????88?
?
-resnet_v1_50/block1/unit_2/bottleneck_v1/ReluRelu,resnet_v1_50/block1/unit_2/bottleneck_v1/add*0
_output_shapes
:?????????88?*
T0
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*%
valueB"         @   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
valueB
 *?d?=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
seed2 *
dtype0*'
_output_shapes
:?@*

seed 
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*'
_output_shapes
:?@
?
Sresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*'
_output_shapes
:?@
?
6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights
VariableV2*
	container *
shape:?@*
dtype0*'
_output_shapes
:?@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weightsSresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
validate_shape(*'
_output_shapes
:?@
?
;resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*'
_output_shapes
:?@
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block1/unit_2/bottleneck_v1/Relu;resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????88@
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes
:@*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
valueB@*  ??
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:@
?
Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes
:@*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
valueB@*    
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:@
?
Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes
:@
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
Hresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes
:@
?
Gresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*/
_output_shapes
:?????????88@
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*%
valueB"      @   @   
?
Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
valueB
 *???=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*&
_output_shapes
:@@*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
Sresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights
VariableV2*
dtype0*&
_output_shapes
:@@*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
	container *
shape:@@
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weightsSresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*&
_output_shapes
:@@
?
;resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*&
_output_shapes
:@@
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Relu;resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*/
_output_shapes
:?????????88@*
	dilations
*
T0
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes
:@*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes
:@*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
valueB@*    
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:@
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes
:@
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB@*    *
dtype0*
_output_shapes
:@
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:@*
dtype0*
_output_shapes
:@
?
Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes
:@*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB@*  ??*
dtype0*
_output_shapes
:@
?
Hresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:@
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
Mresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes
:@
?
Gresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*G
_output_shapes5
3:?????????88@:@:@:@:@*
is_training( 
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*/
_output_shapes
:?????????88@
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*%
valueB"      @      
?
Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
valueB
 *?dN>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
seed2 *
dtype0*'
_output_shapes
:@?*

seed 
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*'
_output_shapes
:@?
?
Sresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights
VariableV2*
dtype0*'
_output_shapes
:@?*
shared_name *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
	container *
shape:@?
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weightsSresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*'
_output_shapes
:@?*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*'
_output_shapes
:@?*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Relu;resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:?????????88?*
	dilations
*
T0
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Yresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:?????????88?:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block1/unit_3/bottleneck_v1/addAdd-resnet_v1_50/block1/unit_2/bottleneck_v1/ReluGresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*0
_output_shapes
:?????????88?*
T0
?
-resnet_v1_50/block1/unit_3/bottleneck_v1/ReluRelu,resnet_v1_50/block1/unit_3/bottleneck_v1/add*0
_output_shapes
:?????????88?*
T0
?
%resnet_v1_50/block1/MaxPool2D/MaxPoolMaxPool-resnet_v1_50/block1/unit_3/bottleneck_v1/Relu*0
_output_shapes
:??????????*
T0*
strides
*
data_formatNHWC*
ksize
*
paddingSAME
?
'resnet_v1_50/block1/strided_slice/stackConst*
dtype0*
_output_shapes
:*%
valueB"                
?
)resnet_v1_50/block1/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*%
valueB"                
?
)resnet_v1_50/block1/strided_slice/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
?
!resnet_v1_50/block1/strided_sliceStridedSliceresnet_v1_50/strided_slice_1'resnet_v1_50/block1/strided_slice/stack)resnet_v1_50/block1/strided_slice/stack_1)resnet_v1_50/block1/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*A
_output_shapes/
-:+???????????????????????????*
Index0*
T0
?
\resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*%
valueB"            
?
[resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/meanConst*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
]resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddevConst*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *?d?=*
dtype0*
_output_shapes
: 
?
fresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal\resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
seed2 
?
Zresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mulMulfresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormal]resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddev*
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normalAddZresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mul[resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mean*
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights
VariableV2*
shared_name *L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/AssignAssign9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weightsVresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*(
_output_shapes
:??
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/readIdentity9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??*
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleConst*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2LossL2Loss>resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/read*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights
?
Sresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizerMulYresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleZresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/Conv2DConv2D%resnet_v1_50/block1/MaxPool2D/MaxPool>resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Rresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/onesConst*T
_classJ
HFloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Aresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *T
_classJ
HFloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
	container *
shape:?
?
Hresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignAssignAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaRresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readIdentityAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
T0*T
_classJ
HFloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
_output_shapes	
:?
?
Rresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zerosConst*S
_classI
GEloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *S
_classI
GEloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
	container *
shape:?
?
Gresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignAssign@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/betaRresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Eresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readIdentity@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
_output_shapes	
:?*
T0*S
_classI
GEloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zerosConst*Z
_classP
NLloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Z
_classP
NLloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
	container *
shape:?
?
Nresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignAssignGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanYresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Lresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readIdentityGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
T0*Z
_classP
NLloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
_output_shapes	
:?
?
\resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*^
_classT
RPloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB?*  ??
?
Kresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *^
_classT
RPloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
	container *
shape:?
?
Rresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignAssignKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance\resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Presnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/readIdentityKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*^
_classT
RPloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormFusedBatchNorm8resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/Conv2DFresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readEresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readLresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readPresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
Aresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*%
valueB"         ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
valueB
 *?d?=
?
cresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weightsSresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Conv2DConv2D%resnet_v1_50/block1/MaxPool2D/MaxPool;resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*
valueB
 *?B=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weightsSresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Relu;resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*%
valueB"      ?      
?
Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
valueB
 *E?>
?
cresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weightsSresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Relu;resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Iresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block2/unit_1/bottleneck_v1/addAddJresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormGresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block2/unit_1/bottleneck_v1/ReluRelu,resnet_v1_50/block2/unit_1/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*%
valueB"         ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weightsSresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block2/unit_1/bottleneck_v1/Relu;resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
valueB
 *?B=
?
cresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weightsSresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Relu;resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*%
valueB"      ?      *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
valueB
 *E?>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weightsSresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Relu;resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Yresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block2/unit_2/bottleneck_v1/addAdd-resnet_v1_50/block2/unit_1/bottleneck_v1/ReluGresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block2/unit_2/bottleneck_v1/ReluRelu,resnet_v1_50/block2/unit_2/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*%
valueB"         ?   
?
Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
valueB
 *E??=
?
cresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weightsSresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block2/unit_2/bottleneck_v1/Relu;resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
valueB
 *?B=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weightsSresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Relu;resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*%
valueB"      ?      
?
Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
valueB
 *E?>*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weightsSresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Relu;resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block2/unit_3/bottleneck_v1/addAdd-resnet_v1_50/block2/unit_2/bottleneck_v1/ReluGresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block2/unit_3/bottleneck_v1/ReluRelu,resnet_v1_50/block2/unit_3/bottleneck_v1/add*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*%
valueB"         ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weightsSresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
<resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block2/unit_3/bottleneck_v1/Relu;resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
?
Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*%
valueB"      ?   ?   *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
valueB
 *?B=
?
cresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weightsSresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Relu;resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*%
valueB"      ?      *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
valueB
 *E?>
?
cresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weightsSresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Relu;resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
?
Cresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta
?
Vresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Iresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Yresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block2/unit_4/bottleneck_v1/addAdd-resnet_v1_50/block2/unit_3/bottleneck_v1/ReluGresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block2/unit_4/bottleneck_v1/ReluRelu,resnet_v1_50/block2/unit_4/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
%resnet_v1_50/block2/MaxPool2D/MaxPoolMaxPool-resnet_v1_50/block2/unit_4/bottleneck_v1/Relu*
strides
*
data_formatNHWC*
ksize
*
paddingSAME*0
_output_shapes
:??????????*
T0
?
'resnet_v1_50/block2/strided_slice/stackConst*%
valueB"                *
dtype0*
_output_shapes
:
?
)resnet_v1_50/block2/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*%
valueB"                
?
)resnet_v1_50/block2/strided_slice/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
?
!resnet_v1_50/block2/strided_sliceStridedSlice!resnet_v1_50/block1/strided_slice'resnet_v1_50/block2/strided_slice/stack)resnet_v1_50/block2/strided_slice/stack_1)resnet_v1_50/block2/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*A
_output_shapes/
-:+???????????????????????????
?
\resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shapeConst*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
[resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/meanConst*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
]resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *E??=
?
fresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal\resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shape*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Zresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mulMulfresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormal]resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddev*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normalAddZresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mul[resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mean*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
	container *
shape:??
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/AssignAssign9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weightsVresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/readIdentity9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *??8
?
Zresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2LossL2Loss>resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/read*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
_output_shapes
: 
?
Sresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizerMulYresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleZresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/Conv2DConv2D%resnet_v1_50/block2/MaxPool2D/MaxPool>resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
bresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/ConstConst*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Rresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/onesFillbresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/shape_as_tensorXresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/Const*
T0*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*

index_type0*
_output_shapes	
:?
?
Aresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
	container *
shape:?
?
Hresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignAssignAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaRresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readIdentityAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
_output_shapes	
:?*
T0*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
bresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB
 *    
?
Rresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zerosFillbresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/shape_as_tensorXresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*

index_type0
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
	container *
shape:?
?
Gresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignAssign@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/betaRresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Eresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readIdentity@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
T0*S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
_output_shapes	
:?
?
iresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB:?
?
_resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/ConstConst*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zerosFilliresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor_resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*

index_type0
?
Gresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
	container *
shape:?
?
Nresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignAssignGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanYresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Lresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readIdentityGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
T0*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
_output_shapes	
:?
?
lresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB:?
?
bresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/ConstConst*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
\resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/onesFilllresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorbresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*

index_type0
?
Kresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
VariableV2*
shared_name *^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Rresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignAssignKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance\resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Presnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/readIdentityKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormFusedBatchNorm8resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/Conv2DFresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readEresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readLresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readPresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
Aresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
<resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Conv2DConv2D%resnet_v1_50/block2/MaxPool2D/MaxPool;resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??	=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
valueB
 *?d?=
?
cresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?
?
Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
fresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?
?
_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block3/unit_1/bottleneck_v1/addAddJresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormGresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
-resnet_v1_50/block3/unit_1/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_1/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block3/unit_1/bottleneck_v1/Relu;resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??	=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
valueB
 *?d?=
?
cresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?
?
Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
fresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0
?
Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
iresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?
?
_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??
?
Yresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block3/unit_2/bottleneck_v1/addAdd-resnet_v1_50/block3/unit_1/bottleneck_v1/ReluGresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
-resnet_v1_50/block3/unit_2/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_2/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
?
<resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block3/unit_2/bottleneck_v1/Relu;resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??	=
?
cresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
?
Sresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
valueB
 *?d?=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*

index_type0
?
=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
fresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
\resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0
?
Hresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block3/unit_3/bottleneck_v1/addAdd-resnet_v1_50/block3/unit_2/bottleneck_v1/ReluGresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block3/unit_3/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_3/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block3/unit_3/bottleneck_v1/Relu;resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
valueB
 *??	=
?
cresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    *
dtype0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
valueB
 *?d?=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*

index_type0
?
=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
fresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
\resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
?
iresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block3/unit_4/bottleneck_v1/addAdd-resnet_v1_50/block3/unit_3/bottleneck_v1/ReluGresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block3/unit_4/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_4/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
valueB
 *?dN=
?
cresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block3/unit_4/bottleneck_v1/Relu;resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma
?
Cresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
valueB
 *??	=
?
cresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
valueB
 *?d?=
?
cresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?
?
Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta
?
fresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean
?
iresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block3/unit_5/bottleneck_v1/addAdd-resnet_v1_50/block3/unit_4/bottleneck_v1/ReluGresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block3/unit_5/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_5/bottleneck_v1/add*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weightsSresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block3/unit_5/bottleneck_v1/Relu;resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    *
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*
valueB
 *??	=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*
seed2 *
dtype0*(
_output_shapes
:??*

seed 
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weightsSresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Relu;resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
valueB
 *?d?=
?
cresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights
?
Sresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights
VariableV2*
shared_name *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weightsSresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
valueB
 *??8
?
Wresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
Presnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Relu;resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma
?
Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*

index_type0
?
=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
fresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0
?
Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?*
dtype0*
_output_shapes
:
?
_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block3/unit_6/bottleneck_v1/addAdd-resnet_v1_50/block3/unit_5/bottleneck_v1/ReluGresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block3/unit_6/bottleneck_v1/ReluRelu,resnet_v1_50/block3/unit_6/bottleneck_v1/add*
T0*0
_output_shapes
:??????????
?
%resnet_v1_50/block3/MaxPool2D/MaxPoolMaxPool-resnet_v1_50/block3/unit_6/bottleneck_v1/Relu*
ksize
*
paddingSAME*0
_output_shapes
:??????????*
T0*
data_formatNHWC*
strides

?
'resnet_v1_50/block3/strided_slice/stackConst*
dtype0*
_output_shapes
:*%
valueB"                
?
)resnet_v1_50/block3/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*%
valueB"                
?
)resnet_v1_50/block3/strided_slice/stack_2Const*%
valueB"            *
dtype0*
_output_shapes
:
?
!resnet_v1_50/block3/strided_sliceStridedSlice!resnet_v1_50/block2/strided_slice'resnet_v1_50/block3/strided_slice/stack)resnet_v1_50/block3/strided_slice/stack_1)resnet_v1_50/block3/strided_slice/stack_2*
end_mask*A
_output_shapes/
-:+???????????????????????????*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask 
?
\resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shapeConst*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
[resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/meanConst*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
]resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddevConst*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
fresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal\resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
seed2 
?
Zresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mulMulfresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/TruncatedNormal]resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/stddev*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normalAddZresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mul[resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal/mean*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
	container *
shape:??
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/AssignAssign9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weightsVresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*(
_output_shapes
:??
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/readIdentity9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*(
_output_shapes
:??
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleConst*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2LossL2Loss>resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/read*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
_output_shapes
: 
?
Sresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizerMulYresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/scaleZresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2DConv2D%resnet_v1_50/block3/MaxPool2D/MaxPool>resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
bresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/ConstConst*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Rresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/onesFillbresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/shape_as_tensorXresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*

index_type0
?
Aresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
Hresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignAssignAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaRresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readIdentityAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
_output_shapes	
:?*
T0*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
bresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/ConstConst*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Rresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zerosFillbresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/shape_as_tensorXresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*

index_type0
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
	container *
shape:?
?
Gresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignAssign@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/betaRresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
Eresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readIdentity@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
_output_shapes	
:?*
T0*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
iresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB:?
?
_resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/ConstConst*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zerosFilliresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor_resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*

index_type0
?
Gresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
	container *
shape:?
?
Nresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignAssignGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanYresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Lresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readIdentityGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
lresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB:?
?
bresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
valueB
 *  ??
?
\resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/onesFilllresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorbresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones/Const*
T0*^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Kresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
	container *
shape:?
?
Rresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignAssignKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance\resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Presnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/readIdentityKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
T0*^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
_output_shapes	
:?
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormFusedBatchNorm8resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/Conv2DFresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/readEresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/readLresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/readPresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
Aresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*%
valueB"            
?
Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
valueB
 *    
?
Zresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
valueB
 *?dN=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weightsSresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2DConv2D%resnet_v1_50/block3/MaxPool2D/MaxPool;resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??<*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights
VariableV2*
	container *
shape:??*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weightsSresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Relu;resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Iresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weightsSresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
_output_shapes
: 
?
<resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Relu;resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?
?
Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0*
_output_shapes	
:?
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*

index_type0
?
=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?
?
fresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?*
dtype0*
_output_shapes
:
?
\resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0*
_output_shapes	
:?
?
Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?
?
_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??
?
Yresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0*
_output_shapes	
:?
?
Hresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Mresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
,resnet_v1_50/block4/unit_1/bottleneck_v1/addAddJresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNormGresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block4/unit_1/bottleneck_v1/ReluRelu,resnet_v1_50/block4/unit_1/bottleneck_v1/add*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
valueB
 *E?=
?
cresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
seed2 
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??
?
6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weightsSresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights
?
Presnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights
?
<resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block4/unit_1/bottleneck_v1/Relu;resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??<*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weightsSresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
?
;resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights*
valueB
 *??8
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
?
Presnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
?
<resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Relu;resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
Bresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*
dtype0*
_output_shapes	
:?*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??
?
Hresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
valueB
 *    
?
Zresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weightsSresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Relu;resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME
?
_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
Cresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?
?
_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?
?
Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
fresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0
?
Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
Iresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?
?
_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Yresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0
?
Hresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( *
epsilon%??'7*
T0
?
>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block4/unit_2/bottleneck_v1/addAdd-resnet_v1_50/block4/unit_1/bottleneck_v1/ReluGresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
-resnet_v1_50/block4/unit_2/bottleneck_v1/ReluRelu,resnet_v1_50/block4/unit_2/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shapeConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*%
valueB"            *
dtype0*
_output_shapes
:
?
Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
valueB
 *E?=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
Sresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/AssignAssign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weightsSresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
;resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/readIdentity6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
Presnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights*
_output_shapes
: 
?
<resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
5resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2DConv2D-resnet_v1_50/block4/unit_2/bottleneck_v1/Relu;resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/onesConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
	container *
shape:?
?
Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gammaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
_output_shapes	
:?
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/betaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanVresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Mresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2DCresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/readBresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/readIresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/ConstConst*
valueB
 *d;?*
dtype0*
_output_shapes
: 
?
3resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/ReluReluGresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*%
valueB"            
?
Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
valueB
 *    
?
Zresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??<
?
cresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
seed2 
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights
?
6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/AssignAssign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weightsSresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
;resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/readIdentity6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/read*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
Presnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer/L2Loss*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
_output_shapes
: 
?
<resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Conv2DConv2D3resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Relu;resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/read*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingSAME
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:?*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
valueB?*  ??
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gammaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Cresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
valueB?*    
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/betaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
Bresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
_output_shapes	
:?
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zerosConst*
dtype0*
_output_shapes	
:?*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
valueB?*    
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
	container *
shape:?
?
Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanVresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
_output_shapes	
:?
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/onesConst*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
valueB?*  ??*
dtype0*
_output_shapes	
:?
?
Hresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
VariableV2*
	container *
shape:?*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Mresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
Gresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Conv2DCresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/readBresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/readIresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
3resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/ReluReluGresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/FusedBatchNorm*0
_output_shapes
:??????????*
T0
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*%
valueB"            
?
Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/meanConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Zresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddevConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*
valueB
 *E??=*
dtype0*
_output_shapes
: 
?
cresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormalYresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*
seed2 
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulMulcresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/TruncatedNormalZresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/stddev*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??
?
Sresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normalAddWresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mulXresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
VariableV2*
dtype0*(
_output_shapes
:??*
shared_name *I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*
	container *
shape:??
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/AssignAssign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weightsSresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
;resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/readIdentity6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*(
_output_shapes
:??*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleConst*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
Wresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2LossL2Loss;resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/read*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
Presnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizerMulVresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/scaleWresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
<resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      
?
5resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2DConv2D3resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Relu;resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
?
_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/ConstConst*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/onesFill_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/shape_as_tensorUresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones/Const*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*

index_type0
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
VariableV2*
shared_name *Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignAssign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gammaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
Cresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readIdentity>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
_output_shapes	
:?*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
?
_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorConst*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB:?*
dtype0*
_output_shapes
:
?
Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/ConstConst*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zerosFill_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/shape_as_tensorUresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros/Const*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta*

index_type0*
_output_shapes	
:?
?
=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta
VariableV2*
shared_name *P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignAssign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/betaOresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
Bresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIdentity=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
_output_shapes	
:?*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
fresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB:?
?
\resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
valueB
 *    
?
Vresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zerosFillfresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/shape_as_tensor\resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros/Const*
_output_shapes	
:?*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*

index_type0
?
Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
VariableV2*
shared_name *W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignAssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanVresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
Iresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readIdentityDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
_output_shapes	
:?
?
iresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensorConst*
dtype0*
_output_shapes
:*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB:?
?
_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/ConstConst*
dtype0*
_output_shapes
: *[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
valueB
 *  ??
?
Yresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/onesFilliresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/shape_as_tensor_resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones/Const*
_output_shapes	
:?*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*

index_type0
?
Hresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
VariableV2*
dtype0*
_output_shapes	
:?*
shared_name *[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
	container *
shape:?
?
Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignAssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceYresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
Mresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/readIdentityHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
_output_shapes	
:?
?
Gresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNormFusedBatchNorm5resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2DCresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/readBresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/readIresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/readMresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read*
epsilon%??'7*
T0*
data_formatNHWC*L
_output_shapes:
8:??????????:?:?:?:?*
is_training( 
?
>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *d;?
?
,resnet_v1_50/block4/unit_3/bottleneck_v1/addAdd-resnet_v1_50/block4/unit_2/bottleneck_v1/ReluGresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm*
T0*0
_output_shapes
:??????????
?
-resnet_v1_50/block4/unit_3/bottleneck_v1/ReluRelu,resnet_v1_50/block4/unit_3/bottleneck_v1/add*0
_output_shapes
:??????????*
T0
?
#resnet_v1_50/AvgPool_1a_7x7/AvgPoolAvgPool-resnet_v1_50/block4/unit_3/bottleneck_v1/Relu*0
_output_shapes
:??????????*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
?
>resnet_v1_50/logits/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@resnet_v1_50/logits/weights*%
valueB"         ?  *
dtype0*
_output_shapes
:
?
=resnet_v1_50/logits/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@resnet_v1_50/logits/weights*
valueB
 *    
?
?resnet_v1_50/logits/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@resnet_v1_50/logits/weights*
valueB
 *E?=*
dtype0*
_output_shapes
: 
?
Hresnet_v1_50/logits/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>resnet_v1_50/logits/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:??*

seed *
T0*.
_class$
" loc:@resnet_v1_50/logits/weights*
seed2 
?
<resnet_v1_50/logits/weights/Initializer/truncated_normal/mulMulHresnet_v1_50/logits/weights/Initializer/truncated_normal/TruncatedNormal?resnet_v1_50/logits/weights/Initializer/truncated_normal/stddev*(
_output_shapes
:??*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights
?
8resnet_v1_50/logits/weights/Initializer/truncated_normalAdd<resnet_v1_50/logits/weights/Initializer/truncated_normal/mul=resnet_v1_50/logits/weights/Initializer/truncated_normal/mean*(
_output_shapes
:??*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights
?
resnet_v1_50/logits/weights
VariableV2*
shared_name *.
_class$
" loc:@resnet_v1_50/logits/weights*
	container *
shape:??*
dtype0*(
_output_shapes
:??
?
"resnet_v1_50/logits/weights/AssignAssignresnet_v1_50/logits/weights8resnet_v1_50/logits/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights
?
 resnet_v1_50/logits/weights/readIdentityresnet_v1_50/logits/weights*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights*(
_output_shapes
:??
?
;resnet_v1_50/logits/kernel/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@resnet_v1_50/logits/weights*
valueB
 *??8*
dtype0*
_output_shapes
: 
?
<resnet_v1_50/logits/kernel/Regularizer/l2_regularizer/L2LossL2Loss resnet_v1_50/logits/weights/read*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights*
_output_shapes
: 
?
5resnet_v1_50/logits/kernel/Regularizer/l2_regularizerMul;resnet_v1_50/logits/kernel/Regularizer/l2_regularizer/scale<resnet_v1_50/logits/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@resnet_v1_50/logits/weights
?
<resnet_v1_50/logits/biases/Initializer/zeros/shape_as_tensorConst*-
_class#
!loc:@resnet_v1_50/logits/biases*
valueB:?*
dtype0*
_output_shapes
:
?
2resnet_v1_50/logits/biases/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *-
_class#
!loc:@resnet_v1_50/logits/biases*
valueB
 *    
?
,resnet_v1_50/logits/biases/Initializer/zerosFill<resnet_v1_50/logits/biases/Initializer/zeros/shape_as_tensor2resnet_v1_50/logits/biases/Initializer/zeros/Const*
T0*-
_class#
!loc:@resnet_v1_50/logits/biases*

index_type0*
_output_shapes	
:?
?
resnet_v1_50/logits/biases
VariableV2*
shared_name *-
_class#
!loc:@resnet_v1_50/logits/biases*
	container *
shape:?*
dtype0*
_output_shapes	
:?
?
!resnet_v1_50/logits/biases/AssignAssignresnet_v1_50/logits/biases,resnet_v1_50/logits/biases/Initializer/zeros*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*-
_class#
!loc:@resnet_v1_50/logits/biases
?
resnet_v1_50/logits/biases/readIdentityresnet_v1_50/logits/biases*
_output_shapes	
:?*
T0*-
_class#
!loc:@resnet_v1_50/logits/biases
r
!resnet_v1_50/logits/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:
?
resnet_v1_50/logits/Conv2DConv2D#resnet_v1_50/AvgPool_1a_7x7/AvgPool resnet_v1_50/logits/weights/read*
paddingSAME*0
_output_shapes
:??????????*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
?
resnet_v1_50/logits/BiasAddBiasAddresnet_v1_50/logits/Conv2Dresnet_v1_50/logits/biases/read*
T0*
data_formatNHWC*0
_output_shapes
:??????????
?
resnet_v1_50/SpatialSqueezeSqueezeresnet_v1_50/logits/BiasAdd*
squeeze_dims
*
T0*(
_output_shapes
:??????????
w
&resnet_v1_50/predictions/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"?????  
?
 resnet_v1_50/predictions/ReshapeReshaperesnet_v1_50/SpatialSqueeze&resnet_v1_50/predictions/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:??????????
?
 resnet_v1_50/predictions/SoftmaxSoftmax resnet_v1_50/predictions/Reshape*(
_output_shapes
:??????????*
T0
y
resnet_v1_50/predictions/ShapeShaperesnet_v1_50/SpatialSqueeze*
_output_shapes
:*
T0*
out_type0
?
"resnet_v1_50/predictions/Reshape_1Reshape resnet_v1_50/predictions/Softmaxresnet_v1_50/predictions/Shape*
T0*
Tshape0*(
_output_shapes
:??????????
b
SoftmaxSoftmaxresnet_v1_50/SpatialSqueeze*(
_output_shapes
:??????????*
T0
m
Pad/paddingsConst*)
value B"               *
dtype0*
_output_shapes

:
e
PadPadSoftmaxPad/paddings*
	Tpaddings0*(
_output_shapes
:??????????*
T0
[
ArgMax/dimensionConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxPadArgMax/dimension*
T0*
output_type0*#
_output_shapes
:?????????*

Tidx0
?
	label_mapConst*I
value@B> B8/grunt/wenmeng.zwm/data/imagenet/imagenet_labelmap.pbtxt*
dtype0*
_output_shapes
: 

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
?
save/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_fe4ebf18407d43a8a2560b48e09dc1f2/part
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
_output_shapes
: *
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
??
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes	
:?*??
value??B???Bglobal_stepB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weightsB!resnet_v1_50/conv1/BatchNorm/betaB"resnet_v1_50/conv1/BatchNorm/gammaB(resnet_v1_50/conv1/BatchNorm/moving_meanB,resnet_v1_50/conv1/BatchNorm/moving_varianceBresnet_v1_50/conv1/weightsBresnet_v1_50/logits/biasesBresnet_v1_50/logits/weights
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
??
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/betaAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/betaAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/betaAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/betaAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gammaDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gammaDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gammaDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights!resnet_v1_50/conv1/BatchNorm/beta"resnet_v1_50/conv1/BatchNorm/gamma(resnet_v1_50/conv1/BatchNorm/moving_mean,resnet_v1_50/conv1/BatchNorm/moving_varianceresnet_v1_50/conv1/weightsresnet_v1_50/logits/biasesresnet_v1_50/logits/weights"/device:CPU:0*?
dtypes?
?2?	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
_output_shapes
: *
T0
??
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes	
:?*??
value??B???Bglobal_stepB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weightsB=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weightsB=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weightsB@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/betaBAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gammaBGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meanBKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_varianceB9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weightsB=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/betaB>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gammaBDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meanBHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_varianceB6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weightsB!resnet_v1_50/conv1/BatchNorm/betaB"resnet_v1_50/conv1/BatchNorm/gammaB(resnet_v1_50/conv1/BatchNorm/moving_meanB,resnet_v1_50/conv1/BatchNorm/moving_varianceBresnet_v1_50/conv1/weightsBresnet_v1_50/logits/biasesBresnet_v1_50/logits/weights
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:?
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	
?
save/AssignAssignglobal_stepsave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0	*
_class
loc:@global_step
?
save/Assign_1Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:1*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
save/Assign_2Assign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:2*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_3AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_4AssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:4*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
save/Assign_5Assign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weightssave/RestoreV2:5*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
?
save/Assign_6Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:6*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
save/Assign_7Assign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:7*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_8AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:8*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_9AssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:9*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
save/Assign_10Assign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weightssave/RestoreV2:10*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights
?
save/Assign_11Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:11*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_12Assign>resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:12*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_13AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:13*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_14AssignHresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:14*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_15Assign6resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weightssave/RestoreV2:15*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights*
validate_shape(*'
_output_shapes
:@?
?
save/Assign_16Assign@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/betasave/RestoreV2:16*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_17AssignAresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gammasave/RestoreV2:17*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma
?
save/Assign_18AssignGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meansave/RestoreV2:18*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_19AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variancesave/RestoreV2:19*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
?
save/Assign_20Assign9resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weightssave/RestoreV2:20*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*'
_output_shapes
:@?
?
save/Assign_21Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:21*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
save/Assign_22Assign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:22*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
save/Assign_23AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:23*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_24AssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:24*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
save/Assign_25Assign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weightssave/RestoreV2:25*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*'
_output_shapes
:?@
?
save/Assign_26Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:26*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
save/Assign_27Assign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:27*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
save/Assign_28AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:28*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
save/Assign_29AssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:29*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_30Assign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weightssave/RestoreV2:30*
validate_shape(*&
_output_shapes
:@@*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights
?
save/Assign_31Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:31*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_32Assign>resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:32*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_33AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:33*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_34AssignHresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:34*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_35Assign6resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weightssave/RestoreV2:35*
validate_shape(*'
_output_shapes
:@?*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights
?
save/Assign_36Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:36*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_37Assign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:37*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_38AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:38*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
save/Assign_39AssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:39*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
save/Assign_40Assign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weightssave/RestoreV2:40*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights*
validate_shape(*'
_output_shapes
:?@
?
save/Assign_41Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:41*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
save/Assign_42Assign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:42*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_43AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:43*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_44AssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:44*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes
:@
?
save/Assign_45Assign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weightssave/RestoreV2:45*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*&
_output_shapes
:@@
?
save/Assign_46Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:46*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_47Assign>resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:47*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_48AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:48*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_49AssignHresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:49*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_50Assign6resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weightssave/RestoreV2:50*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights*
validate_shape(*'
_output_shapes
:@?
?
save/Assign_51Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:51*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_52Assign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:52*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_53AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:53*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_54AssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:54*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_55Assign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weightssave/RestoreV2:55*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights
?
save/Assign_56Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:56*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_57Assign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:57*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_58AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:58*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_59AssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:59*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_60Assign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weightssave/RestoreV2:60*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights
?
save/Assign_61Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:61*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_62Assign>resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:62*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_63AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:63*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_64AssignHresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:64*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_65Assign6resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weightssave/RestoreV2:65*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_66Assign@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/betasave/RestoreV2:66*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_67AssignAresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gammasave/RestoreV2:67*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_68AssignGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meansave/RestoreV2:68*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_69AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variancesave/RestoreV2:69*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance
?
save/Assign_70Assign9resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weightssave/RestoreV2:70*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_71Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:71*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_72Assign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:72*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_73AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:73*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_74AssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:74*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_75Assign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weightssave/RestoreV2:75*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights
?
save/Assign_76Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:76*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
save/Assign_77Assign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:77*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_78AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:78*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_79AssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:79*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_80Assign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weightssave/RestoreV2:80*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_81Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:81*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_82Assign>resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:82*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_83AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:83*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_84AssignHresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:84*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_85Assign6resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weightssave/RestoreV2:85*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights
?
save/Assign_86Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:86*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_87Assign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:87*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_88AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:88*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_89AssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:89*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_90Assign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weightssave/RestoreV2:90*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_91Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:91*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta
?
save/Assign_92Assign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:92*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_93AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:93*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_94AssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:94*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_95Assign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weightssave/RestoreV2:95*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights
?
save/Assign_96Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:96*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_97Assign>resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:97*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_98AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:98*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_99AssignHresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:99*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_100Assign6resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weightssave/RestoreV2:100*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights
?
save/Assign_101Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:101*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_102Assign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:102*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_103AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:103*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_104AssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:104*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_105Assign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weightssave/RestoreV2:105*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights
?
save/Assign_106Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:106*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_107Assign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:107*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_108AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:108*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_109AssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:109*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_110Assign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weightssave/RestoreV2:110*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_111Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:111*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_112Assign>resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:112*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_113AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:113*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_114AssignHresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:114*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_115Assign6resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weightssave/RestoreV2:115*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights
?
save/Assign_116Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:116*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_117Assign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:117*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_118AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:118*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_119AssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:119*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_120Assign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weightssave/RestoreV2:120*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights
?
save/Assign_121Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:121*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_122Assign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:122*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_123AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:123*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_124AssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:124*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_125Assign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weightssave/RestoreV2:125*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights
?
save/Assign_126Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:126*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_127Assign>resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:127*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_128AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:128*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_129AssignHresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:129*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_130Assign6resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weightssave/RestoreV2:130*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights
?
save/Assign_131Assign@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/betasave/RestoreV2:131*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
save/Assign_132AssignAresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gammasave/RestoreV2:132*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_133AssignGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meansave/RestoreV2:133*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
save/Assign_134AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variancesave/RestoreV2:134*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_135Assign9resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weightssave/RestoreV2:135*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_136Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:136*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_137Assign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:137*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_138AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:138*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_139AssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:139*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_140Assign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weightssave/RestoreV2:140*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_141Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:141*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_142Assign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:142*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_143AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:143*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_144AssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:144*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_145Assign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weightssave/RestoreV2:145*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights
?
save/Assign_146Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:146*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_147Assign>resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:147*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_148AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:148*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_149AssignHresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:149*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_150Assign6resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weightssave/RestoreV2:150*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights
?
save/Assign_151Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:151*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_152Assign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:152*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_153AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:153*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_154AssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:154*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_155Assign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weightssave/RestoreV2:155*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights
?
save/Assign_156Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:156*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_157Assign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:157*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_158AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:158*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_159AssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:159*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_160Assign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weightssave/RestoreV2:160*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights
?
save/Assign_161Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:161*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_162Assign>resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:162*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_163AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:163*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_164AssignHresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:164*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_165Assign6resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weightssave/RestoreV2:165*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_166Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:166*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_167Assign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:167*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_168AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:168*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_169AssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:169*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_170Assign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weightssave/RestoreV2:170*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights
?
save/Assign_171Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:171*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_172Assign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:172*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_173AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:173*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_174AssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:174*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_175Assign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weightssave/RestoreV2:175*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_176Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:176*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_177Assign>resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:177*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_178AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:178*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_179AssignHresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:179*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_180Assign6resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weightssave/RestoreV2:180*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights
?
save/Assign_181Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:181*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_182Assign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:182*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_183AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:183*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_184AssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:184*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_185Assign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weightssave/RestoreV2:185*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_186Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:186*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta
?
save/Assign_187Assign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:187*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_188AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:188*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_189AssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:189*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_190Assign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weightssave/RestoreV2:190*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_191Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:191*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_192Assign>resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:192*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_193AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:193*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_194AssignHresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:194*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_195Assign6resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weightssave/RestoreV2:195*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights
?
save/Assign_196Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:196*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_197Assign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:197*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_198AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:198*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_199AssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:199*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_200Assign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weightssave/RestoreV2:200*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_201Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:201*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_202Assign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:202*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_203AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:203*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_204AssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:204*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_205Assign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weightssave/RestoreV2:205*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights
?
save/Assign_206Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:206*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_207Assign>resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:207*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_208AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:208*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_209AssignHresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:209*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_210Assign6resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weightssave/RestoreV2:210*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_211Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:211*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_212Assign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:212*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_213AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:213*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean
?
save/Assign_214AssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:214*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_215Assign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weightssave/RestoreV2:215*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights
?
save/Assign_216Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:216*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_217Assign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:217*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_218AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:218*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_219AssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:219*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_220Assign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weightssave/RestoreV2:220*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_221Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:221*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_222Assign>resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:222*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_223AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:223*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_224AssignHresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:224*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_225Assign6resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weightssave/RestoreV2:225*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_226Assign@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/betasave/RestoreV2:226*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*S
_classI
GEloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta
?
save/Assign_227AssignAresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gammasave/RestoreV2:227*
use_locking(*
T0*T
_classJ
HFloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_228AssignGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_meansave/RestoreV2:228*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Z
_classP
NLloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean
?
save/Assign_229AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variancesave/RestoreV2:229*
use_locking(*
T0*^
_classT
RPloc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_230Assign9resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weightssave/RestoreV2:230*
use_locking(*
T0*L
_classB
@>loc:@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_231Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:231*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_232Assign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:232*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_233AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:233*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_234AssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:234*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_235Assign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weightssave/RestoreV2:235*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_236Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:236*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta
?
save/Assign_237Assign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:237*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma*
validate_shape(*
_output_shapes	
:?
?
save/Assign_238AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:238*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean
?
save/Assign_239AssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:239*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_240Assign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weightssave/RestoreV2:240*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights
?
save/Assign_241Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:241*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_242Assign>resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:242*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_243AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:243*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_244AssignHresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:244*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance*
validate_shape(*
_output_shapes	
:?
?
save/Assign_245Assign6resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weightssave/RestoreV2:245*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_246Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/betasave/RestoreV2:246*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta
?
save/Assign_247Assign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gammasave/RestoreV2:247*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma
?
save/Assign_248AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_meansave/RestoreV2:248*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_249AssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variancesave/RestoreV2:249*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance
?
save/Assign_250Assign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weightssave/RestoreV2:250*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights
?
save/Assign_251Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/betasave/RestoreV2:251*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta*
validate_shape(*
_output_shapes	
:?
?
save/Assign_252Assign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gammasave/RestoreV2:252*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma
?
save/Assign_253AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_meansave/RestoreV2:253*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean*
validate_shape(*
_output_shapes	
:?
?
save/Assign_254AssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variancesave/RestoreV2:254*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance
?
save/Assign_255Assign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weightssave/RestoreV2:255*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights*
validate_shape(*(
_output_shapes
:??
?
save/Assign_256Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/betasave/RestoreV2:256*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*P
_classF
DBloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta
?
save/Assign_257Assign>resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gammasave/RestoreV2:257*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*Q
_classG
ECloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma
?
save/Assign_258AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_meansave/RestoreV2:258*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*W
_classM
KIloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean
?
save/Assign_259AssignHresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variancesave/RestoreV2:259*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*[
_classQ
OMloc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance
?
save/Assign_260Assign6resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weightssave/RestoreV2:260*
validate_shape(*(
_output_shapes
:??*
use_locking(*
T0*I
_class?
=;loc:@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights
?
save/Assign_261Assign!resnet_v1_50/conv1/BatchNorm/betasave/RestoreV2:261*
use_locking(*
T0*4
_class*
(&loc:@resnet_v1_50/conv1/BatchNorm/beta*
validate_shape(*
_output_shapes
:@
?
save/Assign_262Assign"resnet_v1_50/conv1/BatchNorm/gammasave/RestoreV2:262*
use_locking(*
T0*5
_class+
)'loc:@resnet_v1_50/conv1/BatchNorm/gamma*
validate_shape(*
_output_shapes
:@
?
save/Assign_263Assign(resnet_v1_50/conv1/BatchNorm/moving_meansave/RestoreV2:263*
use_locking(*
T0*;
_class1
/-loc:@resnet_v1_50/conv1/BatchNorm/moving_mean*
validate_shape(*
_output_shapes
:@
?
save/Assign_264Assign,resnet_v1_50/conv1/BatchNorm/moving_variancesave/RestoreV2:264*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*?
_class5
31loc:@resnet_v1_50/conv1/BatchNorm/moving_variance
?
save/Assign_265Assignresnet_v1_50/conv1/weightssave/RestoreV2:265*
validate_shape(*&
_output_shapes
:@*
use_locking(*
T0*-
_class#
!loc:@resnet_v1_50/conv1/weights
?
save/Assign_266Assignresnet_v1_50/logits/biasessave/RestoreV2:266*
validate_shape(*
_output_shapes	
:?*
use_locking(*
T0*-
_class#
!loc:@resnet_v1_50/logits/biases
?
save/Assign_267Assignresnet_v1_50/logits/weightssave/RestoreV2:267*
use_locking(*
T0*.
_class$
" loc:@resnet_v1_50/logits/weights*
validate_shape(*(
_output_shapes
:??
?%
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_167^save/Assign_168^save/Assign_169^save/Assign_17^save/Assign_170^save/Assign_171^save/Assign_172^save/Assign_173^save/Assign_174^save/Assign_175^save/Assign_176^save/Assign_177^save/Assign_178^save/Assign_179^save/Assign_18^save/Assign_180^save/Assign_181^save/Assign_182^save/Assign_183^save/Assign_184^save/Assign_185^save/Assign_186^save/Assign_187^save/Assign_188^save/Assign_189^save/Assign_19^save/Assign_190^save/Assign_191^save/Assign_192^save/Assign_193^save/Assign_194^save/Assign_195^save/Assign_196^save/Assign_197^save/Assign_198^save/Assign_199^save/Assign_2^save/Assign_20^save/Assign_200^save/Assign_201^save/Assign_202^save/Assign_203^save/Assign_204^save/Assign_205^save/Assign_206^save/Assign_207^save/Assign_208^save/Assign_209^save/Assign_21^save/Assign_210^save/Assign_211^save/Assign_212^save/Assign_213^save/Assign_214^save/Assign_215^save/Assign_216^save/Assign_217^save/Assign_218^save/Assign_219^save/Assign_22^save/Assign_220^save/Assign_221^save/Assign_222^save/Assign_223^save/Assign_224^save/Assign_225^save/Assign_226^save/Assign_227^save/Assign_228^save/Assign_229^save/Assign_23^save/Assign_230^save/Assign_231^save/Assign_232^save/Assign_233^save/Assign_234^save/Assign_235^save/Assign_236^save/Assign_237^save/Assign_238^save/Assign_239^save/Assign_24^save/Assign_240^save/Assign_241^save/Assign_242^save/Assign_243^save/Assign_244^save/Assign_245^save/Assign_246^save/Assign_247^save/Assign_248^save/Assign_249^save/Assign_25^save/Assign_250^save/Assign_251^save/Assign_252^save/Assign_253^save/Assign_254^save/Assign_255^save/Assign_256^save/Assign_257^save/Assign_258^save/Assign_259^save/Assign_26^save/Assign_260^save/Assign_261^save/Assign_262^save/Assign_263^save/Assign_264^save/Assign_265^save/Assign_266^save/Assign_267^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"?#
regularization_losses?#
?#
6resnet_v1_50/conv1/kernel/Regularizer/l2_regularizer:0
Uresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Uresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Uresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Uresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/kernel/Regularizer/l2_regularizer:0
Rresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/kernel/Regularizer/l2_regularizer:0
7resnet_v1_50/logits/kernel/Regularizer/l2_regularizer:0"??
model_variables????
?
resnet_v1_50/conv1/weights:0!resnet_v1_50/conv1/weights/Assign!resnet_v1_50/conv1/weights/read:029resnet_v1_50/conv1/weights/Initializer/truncated_normal:08
?
$resnet_v1_50/conv1/BatchNorm/gamma:0)resnet_v1_50/conv1/BatchNorm/gamma/Assign)resnet_v1_50/conv1/BatchNorm/gamma/read:025resnet_v1_50/conv1/BatchNorm/gamma/Initializer/ones:0
?
#resnet_v1_50/conv1/BatchNorm/beta:0(resnet_v1_50/conv1/BatchNorm/beta/Assign(resnet_v1_50/conv1/BatchNorm/beta/read:025resnet_v1_50/conv1/BatchNorm/beta/Initializer/zeros:0
?
*resnet_v1_50/conv1/BatchNorm/moving_mean:0/resnet_v1_50/conv1/BatchNorm/moving_mean/Assign/resnet_v1_50/conv1/BatchNorm/moving_mean/read:02<resnet_v1_50/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
.resnet_v1_50/conv1/BatchNorm/moving_variance:03resnet_v1_50/conv1/BatchNorm/moving_variance/Assign3resnet_v1_50/conv1/BatchNorm/moving_variance/read:02?resnet_v1_50/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
resnet_v1_50/logits/weights:0"resnet_v1_50/logits/weights/Assign"resnet_v1_50/logits/weights/read:02:resnet_v1_50/logits/weights/Initializer/truncated_normal:08
?
resnet_v1_50/logits/biases:0!resnet_v1_50/logits/biases/Assign!resnet_v1_50/logits/biases/read:02.resnet_v1_50/logits/biases/Initializer/zeros:08"?t
trainable_variables?t?t
?
resnet_v1_50/conv1/weights:0!resnet_v1_50/conv1/weights/Assign!resnet_v1_50/conv1/weights/read:029resnet_v1_50/conv1/weights/Initializer/truncated_normal:08
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
resnet_v1_50/logits/weights:0"resnet_v1_50/logits/weights/Assign"resnet_v1_50/logits/weights/read:02:resnet_v1_50/logits/weights/Initializer/truncated_normal:08
?
resnet_v1_50/logits/biases:0!resnet_v1_50/logits/biases/Assign!resnet_v1_50/logits/biases/read:02.resnet_v1_50/logits/biases/Initializer/zeros:08"??
	variables????
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
?
resnet_v1_50/conv1/weights:0!resnet_v1_50/conv1/weights/Assign!resnet_v1_50/conv1/weights/read:029resnet_v1_50/conv1/weights/Initializer/truncated_normal:08
?
$resnet_v1_50/conv1/BatchNorm/gamma:0)resnet_v1_50/conv1/BatchNorm/gamma/Assign)resnet_v1_50/conv1/BatchNorm/gamma/read:025resnet_v1_50/conv1/BatchNorm/gamma/Initializer/ones:0
?
#resnet_v1_50/conv1/BatchNorm/beta:0(resnet_v1_50/conv1/BatchNorm/beta/Assign(resnet_v1_50/conv1/BatchNorm/beta/read:025resnet_v1_50/conv1/BatchNorm/beta/Initializer/zeros:0
?
*resnet_v1_50/conv1/BatchNorm/moving_mean:0/resnet_v1_50/conv1/BatchNorm/moving_mean/Assign/resnet_v1_50/conv1/BatchNorm/moving_mean/read:02<resnet_v1_50/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
.resnet_v1_50/conv1/BatchNorm/moving_variance:03resnet_v1_50/conv1/BatchNorm/moving_variance/Assign3resnet_v1_50/conv1/BatchNorm/moving_variance/read:02?resnet_v1_50/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights:0=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
;resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights:0@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Assign@resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/weights/Initializer/truncated_normal:08
?
Cresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma:0Hresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/AssignHresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/read:02Tresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/gamma/Initializer/ones:0
?
Bresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta:0Gresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/AssignGresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/read:02Tresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/beta/Initializer/zeros:0
?
Iresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean:0Nresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/AssignNresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_mean/Initializer/zeros:0
?
Mresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance:0Rresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/AssignRresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/read:02^resnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/BatchNorm/moving_variance/Initializer/ones:0
?
8resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights:0=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Assign=resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/read:02Uresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/weights/Initializer/truncated_normal:08
?
@resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma:0Eresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/AssignEresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/gamma/Initializer/ones:0
?
?resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta:0Dresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/AssignDresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/read:02Qresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/beta/Initializer/zeros:0
?
Fresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean:0Kresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/AssignKresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/read:02Xresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_mean/Initializer/zeros:0
?
Jresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance:0Oresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/AssignOresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/read:02[resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance/Initializer/ones:0
?
resnet_v1_50/logits/weights:0"resnet_v1_50/logits/weights/Assign"resnet_v1_50/logits/weights/read:02:resnet_v1_50/logits/weights/Initializer/truncated_normal:08
?
resnet_v1_50/logits/biases:0!resnet_v1_50/logits/biases/Assign!resnet_v1_50/logits/biases/read:02.resnet_v1_50/logits/biases/Initializer/zeros:08"?t
while_context?s?s
?A
map/while/while_context*map/while/LoopCond:02map/while/Merge:0:map/while/Identity:0Bmap/while/Exit:0Bmap/while/Exit_1:0Bmap/while/Exit_2:0Bmap/while/Exit_3:0Bmap/while/Exit_4:0Bmap/while/Exit_5:0Bmap/while/Exit_6:0Bmap/while/Exit_7:0Bmap/while/Exit_8:0Bmap/while/Exit_9:0J?5
map/TensorArray:0
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
Bmap/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:0
map/TensorArray_1:0
map/TensorArray_2:0
map/TensorArray_3:0
map/TensorArray_4:0
map/TensorArray_5:0
map/strided_slice:0
 map/while/Assert/Assert/data_0:0
map/while/Assert/Const:0
"map/while/Assert_1/Assert/data_0:0
map/while/Assert_1/Const:0
map/while/Cast:0
map/while/Const:0
map/while/Const_1:0
map/while/Enter:0
map/while/Enter_1:0
map/while/Enter_2:0
map/while/Enter_3:0
map/while/Enter_4:0
map/while/Enter_5:0
map/while/Enter_6:0
map/while/Enter_7:0
map/while/Enter_8:0
map/while/Enter_9:0
map/while/Equal/y:0
map/while/Equal:0
map/while/Exit:0
map/while/Exit_1:0
map/while/Exit_2:0
map/while/Exit_3:0
map/while/Exit_4:0
map/while/Exit_5:0
map/while/Exit_6:0
map/while/Exit_7:0
map/while/Exit_8:0
map/while/Exit_9:0
map/while/ExpandDims/dim:0
map/while/ExpandDims:0
map/while/Greater:0
map/while/GreaterEqual/y:0
map/while/GreaterEqual:0
map/while/GreaterEqual_1/y:0
map/while/GreaterEqual_1:0
map/while/Identity:0
map/while/Identity_1:0
map/while/Identity_2:0
map/while/Identity_3:0
map/while/Identity_4:0
map/while/Identity_5:0
map/while/Identity_6:0
map/while/Identity_7:0
map/while/Identity_8:0
map/while/Identity_9:0
map/while/Less/Enter:0
map/while/Less:0
map/while/Less_1:0
map/while/LogicalAnd:0
map/while/LogicalAnd_1:0
map/while/LoopCond:0
map/while/Maximum:0
map/while/Maximum_1:0
map/while/Maximum_2:0
map/while/Maximum_3:0
map/while/Merge:0
map/while/Merge:1
map/while/Merge_1:0
map/while/Merge_1:1
map/while/Merge_2:0
map/while/Merge_2:1
map/while/Merge_3:0
map/while/Merge_3:1
map/while/Merge_4:0
map/while/Merge_4:1
map/while/Merge_5:0
map/while/Merge_5:1
map/while/Merge_6:0
map/while/Merge_6:1
map/while/Merge_7:0
map/while/Merge_7:1
map/while/Merge_8:0
map/while/Merge_8:1
map/while/Merge_9:0
map/while/Merge_9:1
map/while/NextIteration:0
map/while/NextIteration_1:0
map/while/NextIteration_2:0
map/while/NextIteration_3:0
map/while/NextIteration_4:0
map/while/NextIteration_5:0
map/while/NextIteration_6:0
map/while/NextIteration_7:0
map/while/NextIteration_8:0
map/while/NextIteration_9:0
map/while/Rank:0
map/while/Reshape:0
map/while/ResizeBilinear/size:0
map/while/ResizeBilinear:0
map/while/Rint:0
map/while/Rint_1:0
map/while/Shape:0
map/while/Shape_1:0
map/while/Shape_2:0
map/while/Shape_3:0
map/while/Shape_4:0
map/while/Shape_5:0
map/while/Shape_6:0
map/while/Shape_7:0
map/while/Shape_8:0
map/while/Slice:0
map/while/Slice_1:0
map/while/Squeeze:0
map/while/Switch:0
map/while/Switch:1
map/while/Switch_1:0
map/while/Switch_1:1
map/while/Switch_2:0
map/while/Switch_2:1
map/while/Switch_3:0
map/while/Switch_3:1
map/while/Switch_4:0
map/while/Switch_4:1
map/while/Switch_5:0
map/while/Switch_5:1
map/while/Switch_6:0
map/while/Switch_6:1
map/while/Switch_7:0
map/while/Switch_7:1
map/while/Switch_8:0
map/while/Switch_8:1
map/while/Switch_9:0
map/while/Switch_9:1
#map/while/TensorArrayReadV3/Enter:0
%map/while/TensorArrayReadV3/Enter_1:0
map/while/TensorArrayReadV3:0
%map/while/TensorArrayReadV3_1/Enter:0
'map/while/TensorArrayReadV3_1/Enter_1:0
map/while/TensorArrayReadV3_1:0
5map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/map/while/TensorArrayWrite/TensorArrayWriteV3:0
7map/while/TensorArrayWrite_1/TensorArrayWriteV3/Enter:0
1map/while/TensorArrayWrite_1/TensorArrayWriteV3:0
7map/while/TensorArrayWrite_2/TensorArrayWriteV3/Enter:0
1map/while/TensorArrayWrite_2/TensorArrayWriteV3:0
7map/while/TensorArrayWrite_3/TensorArrayWriteV3/Enter:0
1map/while/TensorArrayWrite_3/TensorArrayWriteV3:0
map/while/ToFloat:0
map/while/ToFloat_1:0
map/while/ToFloat_2:0
map/while/ToFloat_3:0
map/while/ToFloat_4:0
map/while/ToInt32:0
map/while/ToInt32_1:0
map/while/ToInt32_2:0
map/while/add/y:0
map/while/add:0
map/while/add_1/y:0
map/while/add_1:0
map/while/concat/axis:0
map/while/concat:0
map/while/cond/Merge:0
map/while/cond/Merge:1
map/while/cond/Switch:0
map/while/cond/Switch:1
map/while/cond/pred_id:0
map/while/cond/switch_f:0
map/while/cond/switch_t:0
map/while/cond/truediv:0
map/while/cond/truediv_1:0
map/while/mul:0
map/while/mul_1:0
map/while/split/split_dim:0
map/while/split:0
map/while/split:1
map/while/split:2
map/while/stack:0
map/while/stack_1/2:0
map/while/stack_1:0
map/while/stack_2/0:0
map/while/stack_2/1:0
map/while/stack_2:0
map/while/stack_3/2:0
map/while/stack_3:0
map/while/strided_slice/stack:0
!map/while/strided_slice/stack_1:0
!map/while/strided_slice/stack_2:0
map/while/strided_slice:0
!map/while/strided_slice_1/stack:0
#map/while/strided_slice_1/stack_1:0
#map/while/strided_slice_1/stack_2:0
map/while/strided_slice_1:0
!map/while/strided_slice_2/stack:0
#map/while/strided_slice_2/stack_1:0
#map/while/strided_slice_2/stack_2:0
map/while/strided_slice_2:0
!map/while/strided_slice_3/stack:0
#map/while/strided_slice_3/stack_1:0
#map/while/strided_slice_3/stack_2:0
map/while/strided_slice_3:0
!map/while/strided_slice_4/stack:0
#map/while/strided_slice_4/stack_1:0
#map/while/strided_slice_4/stack_2:0
map/while/strided_slice_4:0
!map/while/strided_slice_5/stack:0
#map/while/strided_slice_5/stack_1:0
#map/while/strided_slice_5/stack_2:0
map/while/strided_slice_5:0
!map/while/strided_slice_6/stack:0
#map/while/strided_slice_6/stack_1:0
#map/while/strided_slice_6/stack_2:0
map/while/strided_slice_6:0
!map/while/strided_slice_7/stack:0
#map/while/strided_slice_7/stack_1:0
#map/while/strided_slice_7/stack_2:0
map/while/strided_slice_7:0
!map/while/strided_slice_8/stack:0
#map/while/strided_slice_8/stack_1:0
#map/while/strided_slice_8/stack_2:0
map/while/strided_slice_8:0
map/while/sub/y:0
map/while/sub:0
map/while/sub_1/y:0
map/while/sub_1:0
map/while/sub_2/y:0
map/while/sub_2:0
map/while/sub_3/y:0
map/while/sub_3:0
map/while/sub_4/y:0
map/while/sub_4:0
map/while/truediv/Cast:0
map/while/truediv/Cast_1:0
map/while/truediv/y:0
map/while/truediv:0
map/while/truediv_1/Cast:0
map/while/truediv_1/Cast_1:0
map/while/truediv_1/y:0
map/while/truediv_1:0i
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%map/while/TensorArrayReadV3/Enter_1:0L
map/TensorArray_2:05map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0N
map/TensorArray_4:07map/while/TensorArrayWrite_2/TensorArrayWriteV3/Enter:08
map/TensorArray:0#map/while/TensorArrayReadV3/Enter:0<
map/TensorArray_1:0%map/while/TensorArrayReadV3_1/Enter:0N
map/TensorArray_3:07map/while/TensorArrayWrite_1/TensorArrayWriteV3/Enter:0N
map/TensorArray_5:07map/while/TensorArrayWrite_3/TensorArrayWriteV3/Enter:0-
map/strided_slice:0map/while/Less/Enter:0m
Bmap/TensorArrayUnstack_1/TensorArrayScatter/TensorArrayScatterV3:0'map/while/TensorArrayReadV3_1/Enter_1:0Rmap/while/Enter:0Rmap/while/Enter_1:0Rmap/while/Enter_2:0Rmap/while/Enter_3:0Rmap/while/Enter_4:0Rmap/while/Enter_5:0Rmap/while/Enter_6:0Rmap/while/Enter_7:0Rmap/while/Enter_8:0Rmap/while/Enter_9:0Zmap/strided_slice:0b?
?
map/while/cond/cond_textmap/while/cond/pred_id:0map/while/cond/switch_t:0 *?
map/while/ToFloat_1:0
map/while/ToFloat_2:0
map/while/cond/pred_id:0
map/while/cond/switch_t:0
map/while/cond/truediv/Switch:1
!map/while/cond/truediv/Switch_1:1
map/while/cond/truediv:04
map/while/cond/pred_id:0map/while/cond/pred_id:0:
map/while/ToFloat_1:0!map/while/cond/truediv/Switch_1:18
map/while/ToFloat_2:0map/while/cond/truediv/Switch:1b?
?
map/while/cond/cond_text_1map/while/cond/pred_id:0map/while/cond/switch_f:0*?
map/while/ToFloat:0
map/while/ToFloat_2:0
map/while/cond/pred_id:0
map/while/cond/switch_f:0
!map/while/cond/truediv_1/Switch:0
#map/while/cond/truediv_1/Switch_1:0
map/while/cond/truediv_1:0:
map/while/ToFloat_2:0!map/while/cond/truediv_1/Switch:0:
map/while/ToFloat:0#map/while/cond/truediv_1/Switch_1:04
map/while/cond/pred_id:0map/while/cond/pred_id:0
?2
map/while_1/while_context*map/while_1/LoopCond:02map/while_1/Merge:0:map/while_1/Identity:0Bmap/while_1/Exit:0Bmap/while_1/Exit_1:0Bmap/while_1/Exit_2:0Bmap/while_1/Exit_3:0Bmap/while_1/Exit_4:0Bmap/while_1/Exit_5:0J?/
map/TensorArray_2:0
map/TensorArray_3:0
map/TensorArray_4:0
map/TensorArray_5:0
map/TensorArray_6:0
map/TensorArray_7:0
map/TensorArray_8:0
map/TensorArray_9:0
map/strided_slice:0
map/while/Exit_2:0
map/while/Exit_3:0
map/while/Exit_4:0
map/while/Exit_5:0
map/while/Exit_6:0
map/while/Exit_7:0
map/while/Exit_8:0
map/while/Exit_9:0
map/while_1/Enter:0
map/while_1/Enter_1:0
map/while_1/Enter_2:0
map/while_1/Enter_3:0
map/while_1/Enter_4:0
map/while_1/Enter_5:0
map/while_1/Exit:0
map/while_1/Exit_1:0
map/while_1/Exit_2:0
map/while_1/Exit_3:0
map/while_1/Exit_4:0
map/while_1/Exit_5:0
map/while_1/Identity:0
map/while_1/Identity_1:0
map/while_1/Identity_2:0
map/while_1/Identity_3:0
map/while_1/Identity_4:0
map/while_1/Identity_5:0
map/while_1/Less/Enter:0
map/while_1/Less:0
map/while_1/Less_1:0
map/while_1/LogicalAnd:0
map/while_1/LoopCond:0
map/while_1/Merge:0
map/while_1/Merge:1
map/while_1/Merge_1:0
map/while_1/Merge_1:1
map/while_1/Merge_2:0
map/while_1/Merge_2:1
map/while_1/Merge_3:0
map/while_1/Merge_3:1
map/while_1/Merge_4:0
map/while_1/Merge_4:1
map/while_1/Merge_5:0
map/while_1/Merge_5:1
map/while_1/NextIteration:0
map/while_1/NextIteration_1:0
map/while_1/NextIteration_2:0
map/while_1/NextIteration_3:0
map/while_1/NextIteration_4:0
map/while_1/NextIteration_5:0
map/while_1/Pad:0
map/while_1/Pad_1:0
map/while_1/Pad_2:0
map/while_1/Pad_3:0
map/while_1/Shape:0
map/while_1/Shape_1:0
map/while_1/Shape_2:0
map/while_1/Shape_3:0
map/while_1/Switch:0
map/while_1/Switch:1
map/while_1/Switch_1:0
map/while_1/Switch_1:1
map/while_1/Switch_2:0
map/while_1/Switch_2:1
map/while_1/Switch_3:0
map/while_1/Switch_3:1
map/while_1/Switch_4:0
map/while_1/Switch_4:1
map/while_1/Switch_5:0
map/while_1/Switch_5:1
%map/while_1/TensorArrayReadV3/Enter:0
'map/while_1/TensorArrayReadV3/Enter_1:0
map/while_1/TensorArrayReadV3:0
'map/while_1/TensorArrayReadV3_1/Enter:0
)map/while_1/TensorArrayReadV3_1/Enter_1:0
!map/while_1/TensorArrayReadV3_1:0
'map/while_1/TensorArrayReadV3_2/Enter:0
)map/while_1/TensorArrayReadV3_2/Enter_1:0
!map/while_1/TensorArrayReadV3_2:0
'map/while_1/TensorArrayReadV3_3/Enter:0
)map/while_1/TensorArrayReadV3_3/Enter_1:0
!map/while_1/TensorArrayReadV3_3:0
7map/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1map/while_1/TensorArrayWrite/TensorArrayWriteV3:0
9map/while_1/TensorArrayWrite_1/TensorArrayWriteV3/Enter:0
3map/while_1/TensorArrayWrite_1/TensorArrayWriteV3:0
9map/while_1/TensorArrayWrite_2/TensorArrayWriteV3/Enter:0
3map/while_1/TensorArrayWrite_2/TensorArrayWriteV3:0
9map/while_1/TensorArrayWrite_3/TensorArrayWriteV3/Enter:0
3map/while_1/TensorArrayWrite_3/TensorArrayWriteV3:0
map/while_1/add/y:0
map/while_1/add:0
map/while_1/add_1/y:0
map/while_1/add_1:0
map/while_1/stack/values_1:0
map/while_1/stack:0
map/while_1/stack_1/values_1:0
map/while_1/stack_1:0
map/while_1/stack_2/values_1:0
map/while_1/stack_2:0
map/while_1/stack_3/values_1:0
map/while_1/stack_3:0
!map/while_1/strided_slice/stack:0
#map/while_1/strided_slice/stack_1:0
#map/while_1/strided_slice/stack_2:0
map/while_1/strided_slice:0
#map/while_1/strided_slice_1/stack:0
%map/while_1/strided_slice_1/stack_1:0
%map/while_1/strided_slice_1/stack_2:0
map/while_1/strided_slice_1:0
#map/while_1/strided_slice_2/stack:0
%map/while_1/strided_slice_2/stack_1:0
%map/while_1/strided_slice_2/stack_2:0
map/while_1/strided_slice_2:0
#map/while_1/strided_slice_3/stack:0
%map/while_1/strided_slice_3/stack_1:0
%map/while_1/strided_slice_3/stack_2:0
map/while_1/strided_slice_3:0
#map/while_1/strided_slice_4/stack:0
%map/while_1/strided_slice_4/stack_1:0
%map/while_1/strided_slice_4/stack_2:0
map/while_1/strided_slice_4:0
#map/while_1/strided_slice_5/stack:0
%map/while_1/strided_slice_5/stack_1:0
%map/while_1/strided_slice_5/stack_2:0
map/while_1/strided_slice_5:0
#map/while_1/strided_slice_6/stack:0
%map/while_1/strided_slice_6/stack_1:0
%map/while_1/strided_slice_6/stack_2:0
map/while_1/strided_slice_6:0
#map/while_1/strided_slice_7/stack:0
%map/while_1/strided_slice_7/stack_1:0
%map/while_1/strided_slice_7/stack_2:0
map/while_1/strided_slice_7:0
map/while_1/sub:0
map/while_1/sub_1:0
map/while_1/sub_2:0
map/while_1/sub_3:0
map/while_1/sub_4:0
map/while_1/sub_5:0
map/while_1/sub_6:0
map/while_1/sub_7:0
map/while_1/unstack/Enter:0
map/while_1/unstack:0
map/while_1/unstack:1
map/while_1/unstack:2
map/while_1/unstack_1/Enter:0
map/while_1/unstack_1:0
map/while_1/unstack_1:1
map/while_1/unstack_1:2
map/while_1/unstack_2/Enter:0
map/while_1/unstack_2:0
map/while_1/unstack_3/Enter:0
map/while_1/unstack_3:0
map/while_1/zeros/Const:0
#map/while_1/zeros/shape_as_tensor:0
map/while_1/zeros:0
map/while_1/zeros_1/Const:0
%map/while_1/zeros_1/shape_as_tensor:0
map/while_1/zeros_1:0
map/while_1/zeros_2/Const:0
%map/while_1/zeros_2/shape_as_tensor:0
map/while_1/zeros_2:0
map/while_1/zeros_3/Const:0
%map/while_1/zeros_3/shape_as_tensor:0
map/while_1/zeros_3:03
map/while/Exit_7:0map/while_1/unstack_1/Enter:0>
map/TensorArray_3:0'map/while_1/TensorArrayReadV3_1/Enter:0/
map/strided_slice:0map/while_1/Less/Enter:0=
map/while/Exit_2:0'map/while_1/TensorArrayReadV3/Enter_1:0P
map/TensorArray_7:09map/while_1/TensorArrayWrite_1/TensorArrayWriteV3/Enter:01
map/while/Exit_6:0map/while_1/unstack/Enter:0<
map/TensorArray_2:0%map/while_1/TensorArrayReadV3/Enter:0N
map/TensorArray_6:07map/while_1/TensorArrayWrite/TensorArrayWriteV3/Enter:0?
map/while/Exit_5:0)map/while_1/TensorArrayReadV3_3/Enter_1:0>
map/TensorArray_5:0'map/while_1/TensorArrayReadV3_3/Enter:03
map/while/Exit_9:0map/while_1/unstack_3/Enter:0?
map/while/Exit_4:0)map/while_1/TensorArrayReadV3_2/Enter_1:0P
map/TensorArray_9:09map/while_1/TensorArrayWrite_3/TensorArrayWriteV3/Enter:0>
map/TensorArray_4:0'map/while_1/TensorArrayReadV3_2/Enter:03
map/while/Exit_8:0map/while_1/unstack_2/Enter:0?
map/while/Exit_3:0)map/while_1/TensorArrayReadV3_1/Enter_1:0P
map/TensorArray_8:09map/while_1/TensorArrayWrite_2/TensorArrayWriteV3/Enter:0Rmap/while_1/Enter:0Rmap/while_1/Enter_1:0Rmap/while_1/Enter_2:0Rmap/while_1/Enter_3:0Rmap/while_1/Enter_4:0Rmap/while_1/Enter_5:0Zmap/strided_slice:0""
asset_filepaths

label_map:0"?!
resnet_v1_50/_end_points?!
?!
resnet_v1_50/conv1/Relu:0
Lresnet_v1_50/block1/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm:0
5resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block1/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block1/unit_1/bottleneck_v1/Relu:0
5resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block1/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block1/unit_2/bottleneck_v1/Relu:0
5resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block1/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0
/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0
Lresnet_v1_50/block2/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm:0
5resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block2/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block2/unit_1/bottleneck_v1/Relu:0
5resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block2/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block2/unit_2/bottleneck_v1/Relu:0
5resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block2/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block2/unit_3/bottleneck_v1/Relu:0
5resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block2/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0
/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0
Lresnet_v1_50/block3/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm:0
5resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_1/bottleneck_v1/Relu:0
5resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_2/bottleneck_v1/Relu:0
5resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_3/bottleneck_v1/Relu:0
5resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_4/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_4/bottleneck_v1/Relu:0
5resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_5/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_5/bottleneck_v1/Relu:0
5resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block3/unit_6/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0
/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0
Lresnet_v1_50/block4/unit_1/bottleneck_v1/shortcut/BatchNorm/FusedBatchNorm:0
5resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block4/unit_1/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block4/unit_1/bottleneck_v1/Relu:0
5resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block4/unit_2/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block4/unit_2/bottleneck_v1/Relu:0
5resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Relu:0
5resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Relu:0
Iresnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/FusedBatchNorm:0
/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0
/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0
resnet_v1_50/logits/BiasAdd:0"?
cond_context??
?
map/cond/cond_textmap/cond/pred_id:0map/cond/switch_t:0 *?
map/ToFloat_2:0
map/ToFloat_3:0
map/cond/pred_id:0
map/cond/switch_t:0
map/cond/truediv/Switch:1
map/cond/truediv/Switch_1:1
map/cond/truediv:0,
map/ToFloat_3:0map/cond/truediv/Switch:1(
map/cond/pred_id:0map/cond/pred_id:0.
map/ToFloat_2:0map/cond/truediv/Switch_1:1
?
map/cond/cond_text_1map/cond/pred_id:0map/cond/switch_f:0*?
map/ToFloat_1:0
map/ToFloat_3:0
map/cond/pred_id:0
map/cond/switch_f:0
map/cond/truediv_1/Switch:0
map/cond/truediv_1/Switch_1:0
map/cond/truediv_1:0.
map/ToFloat_3:0map/cond/truediv_1/Switch:0(
map/cond/pred_id:0map/cond/pred_id:00
map/ToFloat_1:0map/cond/truediv_1/Switch_1:0"k
global_step\Z
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0"%
saved_model_main_op


group_deps"q
saved_model_assets[*Y
W
+type.googleapis.com/tensorflow.AssetFileDef(

label_map:0imagenet_labelmap.pbtxt*?	
serving_default?	
=
true_image_shape)
true_image_shape:0?????????
A
image8
image:0+???????????????????????????N
resnet_v1_50/conv18
resnet_v1_50/conv1/Relu:0?????????pp@f
resnet_v1_50/block1O
/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0?????????88?&
probs
Pad:0??????????f
resnet_v1_50/block2O
/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0??????????f
resnet_v1_50/block3O
/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0??????????T
resnet_v1_50/logits=
resnet_v1_50/logits/BiasAdd:0??????????f
resnet_v1_50/block4O
/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0??????????U
resnet_v1_50/spatial_squeeze5
resnet_v1_50/SpatialSqueeze:0??????????S

AvgPool_1aE
%resnet_v1_50/AvgPool_1a_7x7/AvgPool:0??????????$
class
ArgMax:0?????????*
predictions
ArgMax:0?????????b
preprocessed_imagesK
*map/TensorArrayStack/TensorArrayGatherV3:0????????????
logits5
resnet_v1_50/SpatialSqueeze:0??????????tensorflow/serving/predict