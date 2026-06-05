#include "bmg_grouped_gemm_mixed_dtype_runner.hpp"

int main(int argc, const char** argv) {
  Options options;
  cutlass::CommandLine cmd(argc, argv);
  bool const has_group_size = cmd.check_cmd_line_flag("g");
  bool const has_mode = cmd.check_cmd_line_flag("mode");

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  if (!has_group_size) {
    options.g = 32;
  }

  if (!has_mode) {
    options.mode = GemmMode::ConvertAndScale;
  }

  if (options.g != 32) {
    std::cerr << "This MXFP4 prototype models pack32 weights, so --g must be 32." << std::endl;
    return -1;
  }

  if (options.mode != GemmMode::ConvertAndScale) {
    std::cerr << "This MXFP4 prototype supports int4 weight + bf16 scale only; use --mode=1." << std::endl;
    return -1;
  }

  if (options.a_narrower) {
    std::cerr << "This MXFP4 prototype supports B/weight as the transformed int4 operand only." << std::endl;
    return -1;
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementWeight = int4_t;
  using ElementActivation = float_e2m1_t;
  using ElementOutput = float;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementScale = bfloat16_t;
  using StrideScale = cute::Stride<_1, int64_t, int64_t>;

  using GmemTiledCopyActivation = void;
  using GmemTiledCopyWeight = XE_2D_U4x32x16_LD_T;

  using TileShape = Shape<_16, _64, _64>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementActivation>>, Layout<TileShape>,
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
          XE_2D_U32x8x16_ST_N,
          void, void>;

  using GemmAdapterBuilder = typename helpers::MixedGemmUniversalAdapterBuilder<ProblemShape, CollectiveEpilogue>;
  using MixedBuilderQuant = helpers::MixedCollectiveMmaBuilder<GEMMDispatchPolicy, TileShape,
                                cutlass::gemm::TagToStrideA_t<LayoutA*>,
                                cutlass::gemm::TagToStrideB_t<LayoutB*>,
                                TiledMma, GmemTiledCopyActivation, GmemTiledCopyWeight>;

  using MainloopConvertAndScale = MixedBuilderQuant::template CollectiveMma<
        ElementActivation, cute::tuple<ElementWeight, ElementScale, StrideScale*>>;
  using GemmConvertAndScale = GemmAdapterBuilder::template GemmUniversalAdapter<MainloopConvertAndScale>;

  std::cout << "Running MXFP4 MMA prototype: A=e2m1, B=int4 pack32 + bf16 scale, mode=ConvertAndScale." << std::endl;
  CUTLASS_CHECK(ExampleRunner<GemmConvertAndScale>{}.run(options, hw_info));
}