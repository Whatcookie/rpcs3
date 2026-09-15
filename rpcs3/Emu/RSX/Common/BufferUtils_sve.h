#pragma once

#include <arm_sve.h>

#if defined(__clang__)
#define BUFFERUTILS_SVE __attribute__((target("sve")))
#else
#define BUFFERUTILS_SVE __attribute__((target("+sve")))
#endif

namespace
{
	template <typename T>
	BUFFERUTILS_SVE std::tuple<T, T, u32> upload_swapped_sve_skip_restart(std::span<to_be_t<const T>> src, std::span<T> dst, T restart_index)
	{
		const u32 count = ::size32(src);
		u32 written = 0;
		auto min = svdup_n_u32(static_cast<T>(-1));
		auto max = svdup_n_u32(0);
		const auto all = svptrue_b32();
		for (u64 i = 0; i < count; i += svcntw())
		{
			const auto active = svwhilelt_b32(i, u64{count});
			svuint32_t value;
			if constexpr (sizeof(T) == 2)
			{
				// SVE COMPACT operates on words or doublewords, so widen the halfwords.
				value = svld1uh_u32(active, reinterpret_cast<const u16*>(src.data()) + i);
				value = svlsr_n_u32_x(active, svrevb_u32_x(active, value), 16);
			}
			else
			{
				value = svld1_u32(active, reinterpret_cast<const u32*>(src.data()) + i);
				value = svrevb_u32_x(active, value);
			}
			const auto keep = svcmpne_n_u32(active, value, restart_index);
			min = svmin_u32_m(keep, min, value);
			max = svmax_u32_m(keep, max, value);
			const auto packed = svcompact_u32(keep, value);
			const u32 processed = svcntp_b32(all, keep);
			const auto output = svwhilelt_b32(u64{0}, u64{processed});
			if constexpr (sizeof(T) == 2)
			{
				svst1h_u32(output, dst.data() + written, packed);
			}
			else
			{
				svst1_u32(output, dst.data() + written, packed);
			}
			written += processed;
		}
		return {static_cast<T>(svminv_u32(all, min)), static_cast<T>(svmaxv_u32(all, max)), written};
	}
}

#undef BUFFERUTILS_SVE
