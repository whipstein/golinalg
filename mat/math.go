package mat

import (
	"math"
	"math/cmplx"
)

const (
	uvnan    = 0xFF800001
	uvinf    = 0x7F800000
	uvneginf = 0xFF800000
	uvone    = 0x3F800000
	mask     = 0xFF
	shift    = 32 - 8 - 1
	bias     = 127
	signMask = 1 << 31
	fracMask = 1<<shift - 1
)

func absf32(a float32) float32 {
	if a < 0 {
		return -a
	}
	return a
}

func absc64(a complex64) float32 {
	return float32(cmplx.Abs(complex128(a)))
}

func absc128(a complex128) float64 {
	return cmplx.Abs(a)
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func abssumc64(a complex64) float32 {
	return float32(absf32(real(a)) + absf32(imag(a)))
}

func abssumc128(a complex128) float64 {
	return (math.Abs(real(a)) + math.Abs(imag(a)))
}

func ceilingf32(a float32) int {
	return int(math.Ceil(float64(a)))
}

func conjc64(a complex64) complex64 {
	return complex64(cmplx.Conj(complex128(a)))
}

func conjc128(a complex128) complex128 {
	return cmplx.Conj(a)
}

func cosf32(a float32) float32 {
	return float32(math.Cos(float64(a)))
}

func epsilonf32() float32 {
	return powf32(2, -23)
	// return float32(1.1920929e-07)
}

func epsilonf64() float64 {
	return powf64(2, -52)
	// return float32(1.1920929e-07)
}

func expf32(a float32) float32 {
	return float32(math.Exp(float64(a)))
}

func logf32(a float32) float32 {
	return float32(math.Log(float64(a)))
}

func log10f32(a float32) float32 {
	return float32(math.Log10(float64(a)))
}

func maxf32(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

func minf32(a, b float32) float32 {
	if a <= b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func modint(a, b int) int {
	return a % b
}

func nintf32(a float32) int {
	return int(math.Round(float64(a)))
}

func powf32(a, b float32) float32 {
	return float32(math.Pow(float64(a), float64(b)))
}

func powf64(a, b float64) float64 {
	return math.Pow(a, b)
}

func pow(a, b int) int {
	return int(math.Pow(float64(a), float64(b)))
}

func roundf32(a float32) float32 {
	var sd int = 8 // significant digits of a float32
	sc := float32(math.Pow10(sd))
	b := int(a * sc)
	return float32(b) / sc
}

func roundf64(a float64) float64 {
	var sd int = 16 // significant digits of a float64
	sc := math.Pow10(sd)
	b := int(a * sc)
	return float64(b) / sc
}

func signf32(a, b float32) float32 {
	return float32(math.Copysign(float64(a), float64(b)))
}

func signint(a, b int) int {
	return int(math.Copysign(float64(a), float64(b)))
}

func sinf32(a float32) float32 {
	return float32(math.Sin(float64(a)))
}

func sqrtf32(a float32) float32 {
	return float32(math.Sqrt(float64(a)))
}

func sqrtf64(a float64) float64 {
	return math.Sqrt(a)
}
