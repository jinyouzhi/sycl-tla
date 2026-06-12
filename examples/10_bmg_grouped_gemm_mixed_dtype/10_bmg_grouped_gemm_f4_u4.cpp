/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Mixed Precision Xe35 Grouped GEMM example with FP4 MMA and uint4 B input.
*/

#include "bmg_grouped_gemm_mixed_dtype_runner.hpp"

int main(int argc, const char** argv) {
  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using ElementTypeA = cutlass::mx_float4_t<cutlass::float_e2m1_t>;

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = typename ElementTypeA::DataType;
  using ElementInputB = uint4_t;
  using ElementOutput = half_t;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementZero = int4_t;
  using ElementScale = bfloat16_t;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;
  using StrideZero = cute::Stride<_8, cute::Stride<_1, int64_t>, int64_t>;

  using GmemTiledCopyA = XE_2D_U4x16x64_LD_N;
  using GmemTiledCopyB = XE_2D_U4x32x16_LD_T;

  using TileShape = Shape<_16, _64, _64>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<typename helpers::MMAOp<ElementInputA>::type>, Layout<TileShape>,
                                    Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16GroupMixedPrecision<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementAccumulator, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC*>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD*>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U16x8x16_ST_N,
          void, void>;

  using GemmAdapterBuilder = typename helpers::MixedGemmUniversalAdapterBuilder<ProblemShape, CollectiveEpilogue>;

  if(options.a_narrower){
    std::cout << "Not support setting A as narrower type for int4 now." << std::endl;
  } else {
    std::cout << "Setting B as narrower type" << std::endl;
    using MixedBuilderQuant = helpers::MixedCollectiveMmaBuilder<GEMMDispatchPolicy, TileShape,
                                  cutlass::gemm::TagToStrideA_t<LayoutA*>,
                                  cutlass::gemm::TagToStrideB_t<LayoutB*>,
                                  TiledMma, GmemTiledCopyA, GmemTiledCopyB>;
    if(options.mode ==  GemmMode::ConvertOnly) {
      std::cout << "Running in ConvertOnly mode." << std::endl;
      using MainloopConvertOnly = MixedBuilderQuant::template CollectiveMma<ElementInputA, cute::tuple<ElementInputB>>;
      using GemmConvertOnly = GemmAdapterBuilder::template GemmUniversalAdapter<MainloopConvertOnly>;
      CUTLASS_CHECK(ExampleRunner<GemmConvertOnly>{}.run(options, hw_info));
    }else if(options.mode == GemmMode::ConvertAndScale){
      std::cout << "Running in ConvertAndScale mode." << std::endl;
      using MainloopConvertAndScale = MixedBuilderQuant::template CollectiveMma<
            ElementInputA, cute::tuple<ElementInputB, ElementScale, StrideScale*>>;
      using GemmConvertAndScale = GemmAdapterBuilder::template GemmUniversalAdapter<MainloopConvertAndScale>;
      CUTLASS_CHECK(ExampleRunner<GemmConvertAndScale>{}.run(options, hw_info));
    }else{
      std::cout << "Running in ConvertAndScaleWithZeroPoint mode." << std::endl;
      using MainloopConvertAndScaleWithZeroPoint = MixedBuilderQuant::template CollectiveMma<
            ElementInputA, cute::tuple<ElementInputB, ElementScale, StrideScale*, ElementZero, StrideZero*>>;
      using GemmConvertAndScaleWithZeroPoint = GemmAdapterBuilder::template GemmUniversalAdapter<MainloopConvertAndScaleWithZeroPoint>;
      CUTLASS_CHECK(ExampleRunner<GemmConvertAndScaleWithZeroPoint>{}.run(options, hw_info));
    }
  }
}