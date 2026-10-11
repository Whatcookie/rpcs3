#pragma once

#include "ProgramStateCache.h"

#include <array>

#ifdef ARCH_X64
#include "immintrin.h"

#if defined(_MSC_VER) && !defined(__clang__)
#define AVX512_ICL_FUNC
#else
#define AVX512_ICL_FUNC __attribute__((__target__("bmi2,avx512f,avx512bw,avx512dq,avx512cd,avx512vl,avx512bitalg,avx512ifma,avx512vbmi,avx512vbmi2,avx512vnni,avx512vpopcntdq")))
#endif

static AVX512_ICL_FUNC NEVER_INLINE usz get_vertex_program_ucode_hash_512(const RSXVertexProgram &program)
{
	// Load all elements of the instruction_mask bitset
	const __m512i* instMask512 = utils::bless<const __m512i>(&program.instruction_mask);
	const __m128i* instMask128 = utils::bless<const __m128i>(&program.instruction_mask);

	const __m512i lowerMask = _mm512_loadu_si512(instMask512);
	const __m128i upper128 = _mm_loadu_si128(instMask128 + 4);
	const __m512i upperMask = _mm512_zextsi128_si512(upper128);

	__m512i maskIndex = _mm512_setzero_si512();
	const __m512i negativeOnes = _mm512_set1_epi64(-1);

	// Special masks to test against bitset
	const __m512i testMask0 = _mm512_set_epi64(
	0x0808080808080808,
	0x0808080808080808,
	0x0404040404040404,
	0x0404040404040404,
	0x0202020202020202,
	0x0202020202020202,
	0x0101010101010101,
	0x0101010101010101);

	const __m512i testMask1 = _mm512_set_epi64(
	0x8080808080808080,
	0x8080808080808080,
	0x4040404040404040,
	0x4040404040404040,
	0x2020202020202020,
	0x2020202020202020,
	0x1010101010101010,
	0x1010101010101010);

	const __m512i* instBuffer = reinterpret_cast<const __m512i*>(program.data.data());
	__m512i acc0 = _mm512_setzero_si512();
	__m512i acc1 = _mm512_setzero_si512();

	__m512i rotMask0 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
	__m512i rotMask1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
	const __m512i rotMaskAdd = _mm512_set_epi64(16, 16, 16, 16, 16, 16, 16, 16);

	u32 instIndex = 0;

	// If there is remainder, add an extra (masked) iteration
	const u32 extraIteration = (program.data.size() % 32 != 0) ? 1 : 0;
	const u32 length = static_cast<u32>(program.data.size() / 32) + extraIteration;

	// The instruction mask will prevent us from reading out of bounds, we do not need a seperate masked loop
	// for the remainder, or a scalar loop.
	while (instIndex < (length))
	{
		const __m512i masks = _mm512_permutex2var_epi8(lowerMask, maskIndex, upperMask);
		const __mmask8 result0 = _mm512_test_epi64_mask(masks, testMask0);
		const __mmask8 result1 = _mm512_test_epi64_mask(masks, testMask1);
		const __m512i load0 = _mm512_maskz_loadu_epi64(result0, (instBuffer + instIndex * 2));
		const __m512i load1 = _mm512_maskz_loadu_epi64(result1, (instBuffer + (instIndex * 2)+ 1));

		const __m512i rotated0 = _mm512_rorv_epi64(load0, rotMask0);
		const __m512i rotated1 = _mm512_rorv_epi64(load1, rotMask1);

		acc0 = _mm512_add_epi64(acc0, rotated0);
		acc1 = _mm512_add_epi64(acc1, rotated1);

		rotMask0 = _mm512_add_epi64(rotMask0, rotMaskAdd);
		rotMask1 = _mm512_add_epi64(rotMask1, rotMaskAdd);
		maskIndex = _mm512_sub_epi8(maskIndex, negativeOnes);

		instIndex++;
	}

	const __m512i result = _mm512_add_epi64(acc0, acc1);
	return _mm512_reduce_add_epi64(result);
}

static AVX512_ICL_FUNC NEVER_INLINE bool vertex_program_compare_512(const RSXVertexProgram &binary1, const RSXVertexProgram &binary2)
	{
		// Load all elements of the instruction_mask bitset
		const __m512i* instMask512 = utils::bless<const __m512i>(&binary1.instruction_mask);
		const __m128i* instMask128 = utils::bless<const __m128i>(&binary1.instruction_mask);

		const __m512i lowerMask = _mm512_loadu_si512(instMask512);
		const __m128i upper128 = _mm_loadu_si128(instMask128 + 4);
		const __m512i upperMask = _mm512_zextsi128_si512(upper128);

		__m512i maskIndex = _mm512_setzero_si512();
		const __m512i negativeOnes = _mm512_set1_epi64(-1);

		// Special masks to test against bitset
		const __m512i testMask0 = _mm512_set_epi64(
		0x0808080808080808,
		0x0808080808080808,
		0x0404040404040404,
		0x0404040404040404,
		0x0202020202020202,
		0x0202020202020202,
		0x0101010101010101,
		0x0101010101010101);

		const __m512i testMask1 = _mm512_set_epi64(
		0x8080808080808080,
		0x8080808080808080,
		0x4040404040404040,
		0x4040404040404040,
		0x2020202020202020,
		0x2020202020202020,
		0x1010101010101010,
		0x1010101010101010);

		const __m512i* instBuffer1 = reinterpret_cast<const __m512i*>(binary1.data.data());
		const __m512i* instBuffer2 = reinterpret_cast<const __m512i*>(binary2.data.data());

		// If there is remainder, add an extra (masked) iteration
		const u32 extraIteration = (binary1.data.size() % 32 != 0) ? 1 : 0;
		const u32 length = static_cast<u32>(binary1.data.size() / 32) + extraIteration;

		u32 instIndex = 0;

		// The instruction mask will prevent us from reading out of bounds, we do not need a seperate masked loop
		// for the remainder, or a scalar loop.
		while (instIndex < (length))
		{
			const __m512i masks = _mm512_permutex2var_epi8(lowerMask, maskIndex, upperMask);

			const __mmask8 result0 = _mm512_test_epi64_mask(masks, testMask0);
			const __mmask8 result1 = _mm512_test_epi64_mask(masks, testMask1);

			const __m512i load0 = _mm512_maskz_loadu_epi64(result0, (instBuffer1 + (instIndex * 2)));
			const __m512i load1 = _mm512_maskz_loadu_epi64(result0, (instBuffer2 + (instIndex * 2)));
			const __m512i load2 = _mm512_maskz_loadu_epi64(result1, (instBuffer1 + (instIndex * 2) + 1));
			const __m512i load3 = _mm512_maskz_loadu_epi64(result1, (instBuffer2 + (instIndex * 2)+ 1));

			const __mmask8 res0 = _mm512_cmpneq_epi64_mask(load0, load1);
			const __mmask8 res1 = _mm512_cmpneq_epi64_mask(load2, load3);

			const u8 result = _kortestz_mask8_u8(res0, res1);

			//kortestz will set result to 1 if all bits are zero, so invert the check for result
			if (!result)
			{
				return false;
			}

			maskIndex = _mm512_sub_epi8(maskIndex, negativeOnes);

			instIndex++;
		}

		return true;
	}

namespace
{
// Entries 0..15 have incoming carry 0; entries 16..31 have incoming carry 1.
// Low 8 bits select 64-bit lanes; bit 8 is the outgoing carry.
constexpr std::array<u16, 32> fragment_hash_slot_masks =
{
	0x00ff, 0x00f3, 0x00cf, 0x00f3, 0x003f, 0x0033, 0x00cf, 0x0033,
	0x01ff, 0x01f3, 0x01cf, 0x01f3, 0x003f, 0x0033, 0x01cf, 0x0033,
	0x00fc, 0x00fc, 0x00cc, 0x00cc, 0x003c, 0x003c, 0x00cc, 0x00cc,
	0x01fc, 0x01fc, 0x01cc, 0x01cc, 0x003c, 0x003c, 0x01cc, 0x01cc
};

constexpr auto fragment_hash_dual_masks = []
{
	std::array<u32, 16> result{};
	for (u32 flags = 0; flags < 16; flags++)
	{
		for (u32 carry = 0; carry < 2; carry++)
		{
			result[flags] |= u32(fragment_hash_slot_masks[flags | (carry << 4)]) << (carry * 16);
		}
	}
	return result;
}();
static_assert(sizeof(fragment_hash_dual_masks) == 64);

constexpr auto fragment_hash_eight_masks = []
{
	std::array<u64, 256> result{};
	for (u32 flags = 0; flags < 256; flags++)
	{
		for (u32 incoming = 0; incoming < 2; incoming++)
		{
			u32 skip = incoming, lanes = 0;
			for (u32 slot = 0; slot < 8; slot++)
			{
				if (skip)
				{
					skip = 0;
				}
				else
				{
					lanes |= 3u << (slot * 2);
					skip = (flags >> slot) & 1;
				}
			}
			result[flags] |= u64(lanes | (skip << 16)) << (incoming * 32);
		}
	}
	return result;
}();
static_assert(sizeof(fragment_hash_eight_masks) == 2048);

AVX512_ICL_FUNC u32 fragment_hash_constant_slots(__m512i inst)
{
	// Isolate the two-bit type field; type 2 denotes an inline constant.
	const auto types = _mm512_and_si512(inst, _mm512_set1_epi32(0x300));
	// One bit per 32-bit word; bits 1..3 of each nibble are sources, excluding bit 0.
	const u32 m = _mm512_cmpeq_epi32_mask(types, _mm512_set1_epi32(0x200));
	// Adding 0b0110 to selected bits 1/2 carries into bit 3 if either is set, without crossing nibbles.
	// OR includes the third source; PEXT gathers bits 3, 7, 11 and 15 into four instruction flags.
	return _pext_u32(((m & 0x6666u) + 0x6666u) | m, 0x8888u);
}

constexpr auto fragment_compare_dual_masks = []
{
	std::array<u32, 16> result{};
	for (u32 flags = 0; flags < 16; flags++)
	{
		for (u32 incoming = 0; incoming < 2; incoming++)
		{
			result[flags] |= u32(fragment_hash_slot_masks[flags | (incoming << 4)]) << (incoming * 16);
		}
	}
	return result;
}();
static_assert(sizeof(fragment_compare_dual_masks) == 64);

constexpr auto fragment_compare_eight_masks = []
{
	std::array<u64, 256> result{};
	for (u32 flags = 0; flags < 256; flags++)
	{
		for (u32 incoming = 0; incoming < 2; incoming++)
		{
			u32 skip = incoming, lanes = 0;
			for (u32 slot = 0; slot < 8; slot++)
			{
				if (skip)
				{
					skip = 0;
				}
				else
				{
					lanes |= 3u << (slot * 2);
					skip = (flags >> slot) & 1;
				}
			}
			result[flags] |= u64(lanes | (skip << 16)) << (incoming * 32);
		}
	}
	return result;
}();
static_assert(sizeof(fragment_compare_eight_masks) == 2048);

AVX512_ICL_FUNC u32 fragment_compare_constant_slots(__m512i inst)
{
	const auto types = _mm512_and_si512(inst, _mm512_set1_epi32(0x300));
	const u32 m = _mm512_cmpeq_epi32_mask(types, _mm512_set1_epi32(0x200));
	return _pext_u32(((m & 0x6666u) + 0x6666u) | m, 0x8888u);
}
}

static AVX512_ICL_FUNC NEVER_INLINE usz get_fragment_program_ucode_hash_512(const RSXFragmentProgram& program)
{
	const usz slots = program.ucode_length / 16;
	if (slots < 4)
	{
		return program_hash_util::fragment_program_utils::get_fragment_program_ucode_hash(program);
	}

	const auto data = static_cast<const u8*>(program.get_data());
	__m512i acc = _mm512_setzero_si512();
	__m512i acc_hi = _mm512_setzero_si512();
	__m512i rotations = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
	const __m512i rotation_step = _mm512_set1_epi64(8);
	u32 carry = 0;
	const usz groups = (slots + 7) / 8;
	const usz tail_slots = slots % 8;
	const u32 tail_live = tail_slots ? (1u << (tail_slots * 2)) - 1 : 0xffff;
	for (usz group = 0, index = 0; group < groups; group++, index += 8)
	{
		const u32 live = group + 1 == groups ? tail_live : 0xffff;
		const usz high_offset = live > 0xff ? 64 : 0;
		const auto ptr = data + index * 16;
		const __m512i lo = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live), ptr);
		const __m512i hi = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live >> 8), ptr + high_offset);
		const u32 flags = fragment_hash_constant_slots(lo) | (fragment_hash_constant_slots(hi) << 4);
		const u32 state = static_cast<u32>(fragment_hash_eight_masks[flags] >> (carry * 32));
		const u32 mask = state & live;
		carry = state >> 16;
		acc = _mm512_mask_add_epi64(acc, static_cast<__mmask8>(mask), acc, _mm512_rorv_epi64(lo, rotations));
		const __m512i hi_rotations = _mm512_add_epi64(rotations, rotation_step);
		acc_hi = _mm512_mask_add_epi64(acc_hi, static_cast<__mmask8>(mask >> 8), acc_hi, _mm512_rorv_epi64(hi, hi_rotations));
		rotations = _mm512_add_epi64(hi_rotations, rotation_step);
	}

	return static_cast<usz>(_mm512_reduce_add_epi64(_mm512_add_epi64(acc, acc_hi)));
}

static AVX512_ICL_FUNC NEVER_INLINE bool compare_fragment_program_ucode_512(const void* lhs, const void* rhs, usz slots)
{
	const auto data1 = static_cast<const u8*>(lhs);
	const auto data2 = static_cast<const u8*>(rhs);
	u32 carry = 0;
	usz index = 0;
	// Preserve a bounded physical four-slot prefix after the unchanged probe.
	if (slots >= 4)
	{
		const __m512i inst1 = _mm512_loadu_si512(data1);
		const __m512i inst2 = _mm512_loadu_si512(data2);
		const u32 flags = fragment_compare_constant_slots(inst1);
		const u16 state = static_cast<u16>(fragment_compare_dual_masks[flags]);
		if (_mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(state), inst1, inst2))
		{
			return false;
		}
		carry = state >> 8;
		index = 4;
	}
	const usz groups = (slots - index + 7) / 8;
	const usz tail_slots = (slots - index) % 8;
	const u32 tail_live = tail_slots ? (1u << (tail_slots * 2)) - 1 : 0xffff;
	for (usz group = 0; group < groups; group++, index += 8)
	{
		const u32 live = group + 1 == groups ? tail_live : 0xffff;
		const usz high_offset = live > 0xff ? 64 : 0;
		const auto ptr1 = data1 + index * 16;
		const auto ptr2 = data2 + index * 16;
		const __m512i lo1 = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live), ptr1);
		const __m512i lo2 = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live), ptr2);
		const __m512i hi1 = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live >> 8), ptr1 + high_offset);
		const __m512i hi2 = _mm512_maskz_loadu_epi64(static_cast<__mmask8>(live >> 8), ptr2 + high_offset);
		const u32 flags = fragment_compare_constant_slots(lo1) | (fragment_compare_constant_slots(hi1) << 4);
		const u32 state = static_cast<u32>(fragment_compare_eight_masks[flags] >> (carry * 32));
		const u32 mask = state & live;
		if (_mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(mask), lo1, lo2) ||
			_mm512_mask_cmpneq_epi64_mask(static_cast<__mmask8>(mask >> 8), hi1, hi2))
		{
			return false;
		}
		carry = state >> 16;
	}

	return true;
}
#undef AVX512_ICL_FUNC
#endif
