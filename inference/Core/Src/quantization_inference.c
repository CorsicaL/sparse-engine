#include "arm_nnfunctions.h"
#include "data.h"
#include "func.h"
#include "stdio.h"
#include "main.h"
#include "utils.h"

int quantization_inference(void) {
    uint32_t start, end;
    
    int32_t conv_count = 0, conv_time = 0;
    int32_t linear_count = 0, linear_time = 0;
    int32_t trans_count = 0, trans_time = 0;
    int32_t softmax_count = 0, softmax_time = 0;
    int32_t norm_count = 0, norm_time = 0;
    int32_t pool_count = 0, pool_time = 0;
    int32_t matmul_count = 0, matmul_time = 0;
    int32_t add_count = 0, add_time = 0;

    cmsis_nn_dims input_dims, output_dims, filter_dims, bias_dims;
    
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_conv_params  conv_params;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_layernorm_params norm_params;
    
    cmsis_nn_per_tensor_quant_params t_quant_params;
    cmsis_nn_per_channel_quant_params c_quant_params;

    cmsis_nn_context ctx;

    static q7_t buf[6976]={0};
    static int32_t conv_mult_use[768]={0};
    static int32_t conv_shift_use[768]={0};
 
    static q7_t section[307200]={0};

    c_quant_params.multiplier=conv_mult_use;
    c_quant_params.shift=conv_shift_use;

    ctx.size = sizeof(buf);
    ctx.buf = buf;

    memcpy(&section,&image,3072);

    start = HAL_GetTick();

    // block: downsample0_0

    input_dims.n=1;
    input_dims.h=32;
    input_dims.w=32;
    input_dims.c=3;

    filter_dims.h=3;
    filter_dims.w=3;

    output_dims.h=32;
    output_dims.w=32;
    output_dims.c=128;

    conv_params.stride.h=1;
    conv_params.stride.w=1;
    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.activation.max=127;
    conv_params.activation.min=-128;
    conv_params.input_offset=-2;
    conv_params.output_offset=-128;
    conv_params.dilation.h=1;
    conv_params.dilation.w=1;

    memcpy(conv_mult_use,mult_0,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_0[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_0,&bias_dims,NULL,&output_dims,&section[176128]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    input_dims.c=128;


    output_dims.h=16;
    output_dims.w=16;

    pool_params.stride.h=2;
    pool_params.stride.w=2;
    pool_params.padding.h=1;
    pool_params.padding.w=1;
    pool_params.activation.min=-128;
    pool_params.activation.max=127;

    arm_maxpool_s8_with_quantization(&ctx,&pool_params,&input_dims,&section[176128],&filter_dims,&output_dims,128,-128,1073741824,1,section);


    end = HAL_GetTick();
    pool_time += (end - start);

    printf("type: pool, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    pool_count++;
    start = end;

    printf("downsample0_0 finished\r\n");

    // block: mv2block0_0

    memcpy(&section[274432],section,32768);

    input_dims.h=16;
    input_dims.w=16;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=256;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_1,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_1[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_1,&bias_dims,bias_1,&output_dims,&section[208896]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    input_dims.c=256;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;
    dw_conv_params.padding.h=1;
    dw_conv_params.padding.w=1;
    dw_conv_params.activation.max=127;
    dw_conv_params.activation.min=-128;
    dw_conv_params.input_offset=128;
    dw_conv_params.output_offset=-128;
    dw_conv_params.dilation.h=1;
    dw_conv_params.dilation.w=1;
    dw_conv_params.ch_mult=1;

    memcpy(conv_mult_use,mult_2,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_2[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[208896],&filter_dims,weight_2,&bias_dims,bias_2,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.output_offset=12;

    memcpy(conv_mult_use,mult_3,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_3[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_3,&bias_dims,bias_3,&output_dims,&section[241664]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[241664],&section[274432],-12,1828018944,0,128,2106830592,-1,0,section,-21,2147483647,0,-128,127,32768);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    printf("mv2block0_0 finished\r\n");

    // block: transformer1_0

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=21;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_4,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_4[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_4,&bias_dims,bias_4,&output_dims,&section[274432]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=44;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_5,176);

    for(int i = 0; i < 44; i++) { conv_shift_use[i] = shift_5[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[274432],&filter_dims,weight_5,&bias_dims,bias_5,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    memcpy(&section[295936],section,11264);

    norm_params.activation.max=127;
    norm_params.activation.min=-128;
    norm_params.input_offset=128;
    norm_params.output_offset=-80;

    t_quant_params.multiplier=1277449088;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,44,weight_6,bias_6,section,section);


    end = HAL_GetTick();
    norm_time += (end - start);

    printf("type: norm, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    norm_count++;
    start = end;

    fc_params.activation.max=127;
    fc_params.activation.min=-128;
    fc_params.input_offset=80;
    fc_params.output_offset=-1;

    t_quant_params.multiplier=1140618240;

    input_dims.n=256;

    filter_dims.n=44;

    output_dims.c=132;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_7,&bias_dims,NULL,&output_dims,&section[262144]);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,256,12,11,&section[262144],section);


    end = HAL_GetTick();
    trans_time += (end - start);

    printf("type: trans, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    trans_count++;
    start = end;

    memcpy(&section[262144],section,33792);

    arm_nn_batch_mat_mult_nt_t_s8(&section[262144],&section[273408],NULL,section,1594825344,-7,256,11,256,1,1,-6,4,-128,127);


    end = HAL_GetTick();
    matmul_time += (end - start);

    printf("type: matmul, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,1024,256,1987060224,21,-248,section);


    end = HAL_GetTick();
    softmax_time += (end - start);

    printf("type: softmax, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[284672],NULL,&section[262144],1365490816,-6,256,256,11,128,1,2,4,-128,127);


    end = HAL_GetTick();
    matmul_time += (end - start);

    printf("type: matmul, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,4,256,11,&section[262144],&section[284672]);


    end = HAL_GetTick();
    trans_time += (end - start);

    printf("type: trans, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    trans_count++;
    start = end;

    fc_params.input_offset=-2;
    fc_params.output_offset=13;

    t_quant_params.multiplier=1121614208;



    output_dims.c=44;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[284672],&filter_dims,weight_8,&bias_dims,bias_8,&output_dims,section);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[295936],-13,1624764800,-2,128,2117586560,0,0,&section[11264],-104,2147483647,0,-128,127,11264);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    memcpy(section,&section[11264],11264);

    norm_params.input_offset=104;
    norm_params.output_offset=-34;

    t_quant_params.multiplier=2062667392;
    t_quant_params.shift=-9;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,256,44,weight_9,bias_9,section,section);


    end = HAL_GetTick();
    norm_time += (end - start);

    printf("type: norm, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    norm_count++;
    start = end;

    memcpy(&section[295936],section,11264);

    fc_params.input_offset=34;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1357418368;
    t_quant_params.shift=-7;



    output_dims.c=256;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_10,&bias_dims,bias_10,&output_dims,&section[230400]);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-1;

    t_quant_params.multiplier=1734779264;
    t_quant_params.shift=-10;


    filter_dims.n=256;

    output_dims.c=44;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[230400],&filter_dims,weight_11,&bias_dims,bias_11,&output_dims,section);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[295936],1,1400422272,-1,34,2003867264,0,0,&section[11264],-33,2147483647,0,-128,127,11264);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    memcpy(section,&section[11264],11264);

    input_dims.n=1;
    input_dims.c=44;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=33;

    memcpy(conv_mult_use,mult_12,176);

    for(int i = 0; i < 44; i++) { conv_shift_use[i] = shift_12[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_12,&bias_dims,bias_12,&output_dims,&section[295936]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_13,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_13[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[295936],&filter_dims,weight_13,&bias_dims,bias_13,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    printf("transformer1_0 finished\r\n");

    // block: downsample1_0

    input_dims.c=128;


    output_dims.c=256;


    memcpy(conv_mult_use,mult_14,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_14[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_14,&bias_dims,bias_14,&output_dims,&section[241664]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    input_dims.c=256;

    filter_dims.h=3;
    filter_dims.w=3;

    output_dims.h=8;
    output_dims.w=8;

    dw_conv_params.stride.h=2;
    dw_conv_params.stride.w=2;

    memcpy(conv_mult_use,mult_15,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_15[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[241664],&filter_dims,weight_15,&bias_dims,bias_15,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    input_dims.h=8;
    input_dims.w=8;

    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.output_offset=3;

    memcpy(conv_mult_use,mult_16,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_16[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_16,&bias_dims,bias_16,&output_dims,&section[299008]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    memcpy(section,&section[299008],8192);

    printf("downsample1_0 finished\r\n");

    // block: mv2block1_0

    memcpy(&section[299008],section,8192);

    input_dims.c=128;


    output_dims.c=384;

    conv_params.input_offset=-3;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_17,1536);

    for(int i = 0; i < 384; i++) { conv_shift_use[i] = shift_17[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_17,&bias_dims,bias_17,&output_dims,&section[274432]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    input_dims.c=384;

    filter_dims.h=3;
    filter_dims.w=3;


    dw_conv_params.stride.h=1;
    dw_conv_params.stride.w=1;

    memcpy(conv_mult_use,mult_18,1536);

    for(int i = 0; i < 384; i++) { conv_shift_use[i] = shift_18[i]; }

    arm_depthwise_conv_s8(&ctx,&dw_conv_params,&c_quant_params,&input_dims,&section[274432],&filter_dims,weight_18,&bias_dims,bias_18,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=128;

    conv_params.input_offset=128;
    conv_params.output_offset=-1;

    memcpy(conv_mult_use,mult_19,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_19[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_19,&bias_dims,bias_19,&output_dims,&section[290816]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(&section[290816],&section[299008],1,1488835584,0,-3,1525423232,0,0,section,2,2147483647,0,-128,127,8192);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    printf("mv2block1_0 finished\r\n");

    // block: transformer1_0

    input_dims.c=128;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=-2;
    conv_params.output_offset=-128;

    memcpy(conv_mult_use,mult_20,512);

    for(int i = 0; i < 128; i++) { conv_shift_use[i] = shift_20[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_20,&bias_dims,bias_20,&output_dims,&section[299008]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=192;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_21,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_21[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[299008],&filter_dims,weight_21,&bias_dims,bias_21,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    memcpy(&section[294912],section,12288);

    norm_params.input_offset=128;
    norm_params.output_offset=-96;

    t_quant_params.multiplier=1181701120;
    t_quant_params.shift=-8;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,192,weight_22,bias_22,section,section);


    end = HAL_GetTick();
    norm_time += (end - start);

    printf("type: norm, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    norm_count++;
    start = end;

    fc_params.input_offset=96;
    fc_params.output_offset=-3;

    t_quant_params.multiplier=1219765248;
    t_quant_params.shift=-9;

    input_dims.n=64;

    filter_dims.n=192;

    output_dims.c=576;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_23,&bias_dims,NULL,&output_dims,&section[258048]);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,64,12,48,&section[258048],section);


    end = HAL_GetTick();
    trans_time += (end - start);

    printf("type: trans, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    trans_count++;
    start = end;

    memcpy(&section[258048],section,36864);

    arm_nn_batch_mat_mult_nt_t_s8(&section[258048],&section[270336],NULL,section,1790788864,-9,64,48,64,3,3,-8,4,-128,127);


    end = HAL_GetTick();
    matmul_time += (end - start);

    printf("type: matmul, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    matmul_count++;
    start = end;

    arm_softmax_s8_fast(&ctx,section,256,64,1538377856,23,-248,section);


    end = HAL_GetTick();
    softmax_time += (end - start);

    printf("type: softmax, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    softmax_count++;
    start = end;

    arm_nn_batch_mat_mult_s8(&ctx,section,&section[282624],NULL,&section[16384],1090957184,-6,64,64,48,128,3,5,4,-128,127);


    end = HAL_GetTick();
    matmul_time += (end - start);

    printf("type: matmul, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    matmul_count++;
    start = end;

    arm_nn_transpose_bhwc_to_bwhc_q7(1,4,64,48,&section[16384],&section[282624]);


    end = HAL_GetTick();
    trans_time += (end - start);

    printf("type: trans, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    trans_count++;
    start = end;

    fc_params.input_offset=-5;
    fc_params.output_offset=-17;

    t_quant_params.multiplier=1270578688;



    output_dims.c=192;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[282624],&filter_dims,weight_24,&bias_dims,bias_24,&output_dims,section);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[294912],17,1540109056,-1,128,1884994816,0,0,&section[12288],-90,2147483647,0,-128,127,12288);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    memcpy(section,&section[12288],12288);

    norm_params.input_offset=90;
    norm_params.output_offset=-50;

    t_quant_params.multiplier=1977929088;

    arm_layernorm_s8(&ctx,&norm_params,&t_quant_params,64,192,weight_25,bias_25,section,section);


    end = HAL_GetTick();
    norm_time += (end - start);

    printf("type: norm, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    norm_count++;
    start = end;

    memcpy(&section[294912],section,12288);

    fc_params.input_offset=50;
    fc_params.output_offset=-128;

    t_quant_params.multiplier=1742211072;
    t_quant_params.shift=-8;



    output_dims.c=128;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_26,&bias_dims,bias_26,&output_dims,&section[286720]);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    fc_params.input_offset=128;
    fc_params.output_offset=-15;

    t_quant_params.multiplier=1107692544;
    t_quant_params.shift=-9;


    filter_dims.n=128;

    output_dims.c=192;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,&section[286720],&filter_dims,weight_27,&bias_dims,bias_27,&output_dims,section);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    arm_elementwise_add_s8_with_neg(section,&section[294912],15,1272834688,-1,50,2028565888,0,0,&section[12288],-42,2147483647,0,-128,127,12288);


    end = HAL_GetTick();
    add_time += (end - start);

    printf("type: add, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    add_count++;
    start = end;

    memcpy(section,&section[12288],12288);

    input_dims.n=1;
    input_dims.c=192;

    filter_dims.h=3;
    filter_dims.w=3;


    conv_params.padding.h=1;
    conv_params.padding.w=1;
    conv_params.input_offset=42;

    memcpy(conv_mult_use,mult_28,768);

    for(int i = 0; i < 192; i++) { conv_shift_use[i] = shift_28[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_28,&bias_dims,bias_28,&output_dims,&section[294912]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;


    filter_dims.h=1;
    filter_dims.w=1;

    output_dims.c=256;

    conv_params.padding.h=0;
    conv_params.padding.w=0;
    conv_params.input_offset=128;

    memcpy(conv_mult_use,mult_29,1024);

    for(int i = 0; i < 256; i++) { conv_shift_use[i] = shift_29[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,&section[294912],&filter_dims,weight_29,&bias_dims,bias_29,&output_dims,section);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    printf("transformer1_0 finished\r\n");

    // block: last conv

    input_dims.c=256;


    output_dims.c=768;


    memcpy(conv_mult_use,mult_30,3072);

    for(int i = 0; i < 768; i++) { conv_shift_use[i] = shift_30[i]; }

    arm_convolve_s8(&ctx,&conv_params,&c_quant_params,&input_dims,section,&filter_dims,weight_30,&bias_dims,bias_30,&output_dims,&section[258048]);


    end = HAL_GetTick();
    conv_time += (end - start);

    printf("type: conv, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    conv_count++;
    start = end;

    memcpy(section,&section[258048],49152);

    printf("last_conv finished\r\n");

    // block: qglobal_pooling

    input_dims.c=768;

    filter_dims.h=8;
    filter_dims.w=8;

    output_dims.h=1;
    output_dims.w=1;

    pool_params.stride.h=8;
    pool_params.stride.w=8;
    pool_params.padding.h=0;
    pool_params.padding.w=0;

    arm_avgpool_s8_with_quantization(&ctx,&pool_params,&input_dims,section,&filter_dims,&output_dims,128,-128,1087822670,3,&section[306432]);


    end = HAL_GetTick();
    pool_time += (end - start);

    printf("type: pool, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    pool_count++;
    start = end;

    memcpy(section,&section[306432],768);

    printf("qglobal_pooling finished\r\n");

    // block: classifier

    fc_params.output_offset=-48;

    t_quant_params.multiplier=1260552192;
    t_quant_params.shift=-12;


    filter_dims.n=768;

    output_dims.c=10;

    arm_fully_connected_s8(&ctx,&fc_params,&t_quant_params,&input_dims,section,&filter_dims,weight_31,&bias_dims,NULL,&output_dims,&section[307190]);


    end = HAL_GetTick();
    linear_time += (end - start);

    printf("type: linear, time: %d, params: {", end - start);
    print_cmsis_nn_dims(&input_dims, "input_dims");
    printf(", ");
    print_cmsis_nn_dims(&output_dims, "output_dims");
    printf(", ");
    print_cmsis_nn_dims(&filter_dims, "filter_dims");
    printf("}\n");

    linear_count++;
    start = end;

    memcpy(section,&section[307190],10);

    printf("classifier finished\r\n");

    result_check_statistics(&ctx,section,conv_count,conv_time,linear_count,linear_time,trans_count,trans_time,softmax_count,softmax_time,norm_count,norm_time,pool_count,pool_time,matmul_count,matmul_time,add_count,add_time,3);

    return 0;
}
