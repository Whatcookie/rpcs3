#pragma once

#include <arm_neon.h>

namespace
{
template <typename T>
constexpr T neon_index_limit()
{
	return -1;
}

template <typename T>
static inline T neon_min_max(T& min, T& max, T value)
{
	if (value < min)
		min = value;

	if (value > max)
		max = value;

	return value;
}

static inline u64 upload_u16_swapped_neon_restart(const be_t<u16>* src, u16* dst, u32 count, u16 restart_index)
{
	const uint16x8_t restart = vdupq_n_u16(restart_index);
	const uint16x8_t ones = vdupq_n_u16(neon_index_limit<u16>());
	const uint16x8_t zero = vdupq_n_u16(0);
	uint16x8_t min = ones;
	uint16x8_t max = zero;

	u32 i = 0;
	for (; i + 8 <= count; i += 8)
	{
		const uint16x8_t value = vreinterpretq_u16_u8(vrev16q_u8(vld1q_u8(reinterpret_cast<const u8*>(src + i))));
		const uint16x8_t is_restart = vceqq_u16(value, restart);
		const uint16x8_t store_value = vbslq_u16(is_restart, ones, value);
		const uint16x8_t max_value = vbslq_u16(is_restart, zero, value);

		min = vminq_u16(min, store_value);
		max = vmaxq_u16(max, max_value);
		vst1q_u16(dst + i, store_value);
	}

	alignas(16) u16 min_lanes[8];
	alignas(16) u16 max_lanes[8];
	vst1q_u16(min_lanes, min);
	vst1q_u16(max_lanes, max);

	u16 min_index = neon_index_limit<u16>();
	u16 max_index = 0;

	for (u32 lane = 0; lane < 8; lane++)
	{
		if (min_lanes[lane] < min_index)
			min_index = min_lanes[lane];
		if (max_lanes[lane] > max_index)
			max_index = max_lanes[lane];
	}

	for (; i < count; i++)
	{
		const u16 index = src[i].value();
		dst[i] = index == restart_index ? neon_index_limit<u16>() : neon_min_max(min_index, max_index, index);
	}

	return (u64{max_index} << 32) | u64{min_index};
}

static inline u64 upload_u32_swapped_neon_restart(const be_t<u32>* src, u32* dst, u32 count, u32 restart_index)
{
	const uint32x4_t restart = vdupq_n_u32(restart_index);
	const uint32x4_t ones = vdupq_n_u32(neon_index_limit<u32>());
	const uint32x4_t zero = vdupq_n_u32(0);
	uint32x4_t min = ones;
	uint32x4_t max = zero;

	u32 i = 0;
	for (; i + 4 <= count; i += 4)
	{
		const uint32x4_t value = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(reinterpret_cast<const u8*>(src + i))));
		const uint32x4_t is_restart = vceqq_u32(value, restart);
		const uint32x4_t store_value = vbslq_u32(is_restart, ones, value);
		const uint32x4_t max_value = vbslq_u32(is_restart, zero, value);

		min = vminq_u32(min, store_value);
		max = vmaxq_u32(max, max_value);
		vst1q_u32(dst + i, store_value);
	}

	alignas(16) u32 min_lanes[4];
	alignas(16) u32 max_lanes[4];
	vst1q_u32(min_lanes, min);
	vst1q_u32(max_lanes, max);

	u32 min_index = neon_index_limit<u32>();
	u32 max_index = 0;

	for (u32 lane = 0; lane < 4; lane++)
	{
		if (min_lanes[lane] < min_index)
			min_index = min_lanes[lane];
		if (max_lanes[lane] > max_index)
			max_index = max_lanes[lane];
	}

	for (; i < count; i++)
	{
		const u32 index = src[i].value();
		dst[i] = index == restart_index ? neon_index_limit<u32>() : neon_min_max(min_index, max_index, index);
	}

	return (u64{max_index} << 32) | u64{min_index};
}
}
