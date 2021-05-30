package eig

import (
	"fmt"
	"golinalg/golapack"
	"math"
	"testing"
)

// Zchkbl tests ZGEBAL, a routine for balancing a general complex
// matrix and isolating some of its eigenvalues.
func Zchkbl(t *testing.T) {
	var rmax, sfmin, temp, vmax, zero float64
	var _i, i, ihi, ihiin, ilo, iloin, info, j, knt, lda, n, ninfo int
	// dummy := vf(1)
	scale := vf(20)
	scalin := vf(20)
	lmax := make([]int, 3)
	a := cmf(20, 20, opts)
	ain := cmf(20, 20, opts)

	lda = 20
	zero = 0.0

	lmax[0] = 0
	lmax[1] = 0
	lmax[2] = 0
	ninfo = 0
	knt = 0
	rmax = zero
	vmax = zero
	sfmin = golapack.Dlamch(SafeMinimum)
	// meps = golapack.Dlamch(Epsilon)

	nlist := []int{5, 5, 5, 4, 6, 5, 4, 4, 5, 6, 7, 5, 6}
	alist := [][]complex128{
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.30000e+01 + 0.30000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.40000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.50000e+01 + 0.50000e+01i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.10000e+01i, 0.20000e+01 + 0.20000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.10000e+01i, 0.20000e+01 + 0.20000e+01i, 0.30000e+01 + 0.30000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.10000e+01i, 0.20000e+01 + 0.20000e+01i, 0.30000e+01 + 0.30000e+01i, 0.40000e+01 + 0.40000e+01i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.10000e+01i, 0.20000e+01 + 0.20000e+01i, 0.30000e+01 + 0.30000e+01i, 0.40000e+01 + 0.40000e+01i, 0.50000e+01 + 0.50000e+01i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i,
		},
		{
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.10000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.20000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+00 + 0.00000e+00i,
			0.10000e+03 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+03 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10240e+04 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.12800e+03 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.10000e+01i, 0.30000e+04 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.12800e+03i, 0.40000e+01 + 0.00000e+00i, 0.40000e-02 + 0.00000e+00i, 0.50000e+01 + 0.00000e+00i, 0.60000e+03 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.20000e-02i, 0.20000e+01 + 0.00000e+00i,
			0.80000e+01 + 0.00000e+00i, 0.00000e+00 + 0.81920e+04i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.10000e+01i, 0.81920e+04 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.40000e+01 + 0.00000e+00i,
			0.25000e-03 + 0.00000e+00i, 0.12500e-03 + 0.00000e+00i, 0.40000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.64000e+02 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.10240e+04 + 0.10240e+01i, 0.40000e+01 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.81920e+04i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.10000e+07 + 0.00000e+00i, 0.10000e+07 + 0.00000e+00i, 0.10000e+07 + 0.00000e+00i,
			-.20000e+07 + 0.00000e+00i, 0.30000e+01 + 0.10000e+01i, 0.20000e-05 + 0.00000e+00i, 0.30000e-05 + 0.00000e+00i,
			-.30000e+07 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e-05 + 0.10000e+01i, 0.20000e+01 + 0.00000e+00i,
			0.10000e+07 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.30000e-05 + 0.00000e+00i, 0.40000e+07 + 0.10000e+01i,
		},
		{
			0.10000e+01 + 0.00000e+00i, 0.00000e+00 + 0.10000e+05i, 0.00000e+00 + 0.10000e+05i, 0.00000e+00 + 0.10000e+05i,
			-.20000e+05 + 0.00000e+00i, 0.30000e+01 + 0.00000e+00i, 0.20000e-02 + 0.00000e+00i, 0.30000e-02 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, -.30000e+05 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+05 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.00000e+00i, 0.51200e+03 + 0.00000e+00i, 0.40960e+04 + 0.00000e+00i, 0.32768e+05 + 0.00000e+00i, 2.62144e+05 + 0.00000e+00i,
			0.80000e+01 + 0.80000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.80000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.80000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.80000e+01i, 0.00000e+00 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
		},
		{
			0.60000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.40000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.25000e-03 + 0.00000e+00i, 0.12500e-01 + 0.00000e+00i, 0.20000e-01 + 0.00000e+00i, 0.12500e+00 + 0.00000e+00i,
			0.10000e+01 + 0.00000e+00i, 0.12800e+03 + 0.00000e+00i, 0.64000e+02 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, -.20000e+01 + 0.00000e+00i, 0.16000e+02 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.16384e+05 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, -.40000e+03 + 0.00000e+00i, 0.25600e+03 + 0.00000e+00i, -.40000e+04 + 0.00000e+00i,
			-.20000e+01 + 0.00000e+00i, -.25600e+03 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.12500e-01 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.32000e+02 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.40000e-02 + 0.00000e+00i, 0.12500e+00 + 0.00000e+00i, -.20000e+00 + 0.00000e+00i, 0.30000e+01 + 0.00000e+00i,
		},
		{
			0.10000e+04 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.30000e+01 + 0.00000e+00i, 0.40000e+01 + 0.00000e+00i, 0.50000e+06 + 0.00000e+00i,
			0.90000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e-03 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.30000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, -.30000e+03 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i,
			0.90000e+01 + 0.00000e+00i, 0.20000e-02 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, -.10000e+04 + 0.00000e+00i,
			0.60000e+01 + 0.00000e+00i, 0.20000e+03 + 0.00000e+00i, 0.10000e+01 + 0.00000e+00i, 0.60000e+03 + 0.00000e+00i, 0.30000e+01 + 0.00000e+00i,
		},
		{
			1.0000e+00 + 0.0000e+00i, 1.0000e+120 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			1.0000e-120 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+120 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 1.0000e-120 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+120 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e-120 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+120 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e-120 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i, 1.0000e+120 + 0.0000e+00i,
			0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 0.0000e+00 + 0.0000e+00i, 1.0000e-120 + 0.0000e+00i, 1.0000e+00 + 0.0000e+00i,
		},
	}
	iloinlist := []int{1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1}
	ihiinlist := []int{1, 1, 1, 4, 6, 5, 4, 4, 5, 5, 5, 5, 6}
	ainlist := [][]complex128{
		{
			0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.30000e+01 + 0.30000e+01i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.40000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.50000e+01 + 0.50000e+01i,
		},
		{
			0.50000e+01 + 0.50000e+01i, 0.40000e+01 + 0.40000e+01i, 0.30000e+01 + 0.30000e+01i, 0.20000e+01 + 0.20000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.40000e+01 + 0.40000e+01i, 0.30000e+01 + 0.30000e+01i, 0.20000e+01 + 0.20000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.30000e+01 + 0.30000e+01i, 0.20000e+01 + 0.20000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.20000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i,
		},
		{
			0.0000e+00 + 0.00000e+00i, 0.2000e+01 + 0.00000e+00i, 0.3200e+01 + 0.00000e+00i, 0.000e+00 + 0.00000e+00i,
			0.2000e+01 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.3200e+01 + 0.00000e+00i,
			0.3125e+01 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.2000e+01 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 0.3125e+01 + 0.00000e+00i, 0.2000e+01 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i,
		},
		{
			0.50000e+01 + 0.00000e+00i, 0.40000e-02 + 0.00000e+00i, 0.60000e+03 + 0.00000e+00i, 0.00000e+00 + 0.10240e+04i, 0.50000e+00 + 0.00000e+00i, 0.80000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.30000e+04 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.25000e+00 + 0.12500e+00i, 0.20000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.20000e-02i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.20000e+01 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.00000e+00 + 0.00000e+00i, 0.12800e+03 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10240e+04 + 0.00000e+00i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.64000e+02 + 0.00000e+00i, 0.00000e+00 + 0.10240e+04i, 0.20000e+01 + 0.00000e+00i,
		},
		{
			1.0000e+000 + 1.0000e+000i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 250.0000e-003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 2.0000e+000 + 1.0000e+000i, 1.0240e+003 + 0.00000e+00i, 16.0000e+000 + 0.00000e+00i, 16.0000e+000 + 0.00000e+00i,
			256.0000e-003 + 0.00000e+00i, 1.0000e-003 + 0.00000e+00i, 4.0000e+000 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 2.0480e+003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 250.0000e-003 + 0.00000e+00i, 16.0000e+000 + 16.0000e-003i, 4.0000e+000 + 0.00000e+00i, 4.0000e+000 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 2.0480e+003i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 8.0000e+000 + 0.00000e+00i,
		},
		{
			1.0000e+000 + 1.0000e+000i, 1.0000e+006 + 0.00000e+00i, 2.0000e+006 + 0.00000e+00i, 1.0000e+006 + 0.00000e+00i, 250.0000e-003 + 0.00000e+00i,
			-2.0000e+006 + 0.00000e+00i, 3.0000e+000 + 1.0000e+000i, 4.0000e-006 + 0.00000e+00i, 3.0000e-006 + 0.00000e+00i, 16.0000e+000 + 0.00000e+00i,
			-1.5000e+006 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 1.0000e-006 + 1.0000e+000i, 1.0000e+000 + 0.00000e+00i, 2.0480e+003 + 0.00000e+00i,
			1.0000e+006 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 6.0000e-006 + 0.00000e+00i, 4.0000e+006 + 1.0000e+000i, 4.0000e+000 + 0.00000e+00i,
		},
		{
			1.0000e+000 + 0.00000e+00i, 0.0000e-003 + 10.0000e+003i, 0.0000e-003 + 10.0000e+003i, 0.0000e-003 + 5.0000e+003i, 250.0000e-003 + 0.00000e+00i,
			-20.0000e+003 + 0.00000e+00i, 3.0000e+000 + 0.00000e+00i, 2.0000e-003 + 0.00000e+00i, 1.5000e-003 + 0.00000e+00i, 16.0000e+000 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 2.0000e+000 + 1.0000e+000i, 0.0000e-003 + 0.00000e+00i, -15.0000e+003 + 0.00000e+00i, 2.0480e+003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 20.0000e+003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 4.0000e+000 + 0.00000e+00i,
		},
		{
			1.0000e+000 + 0.00000e+00i, 64.0000e+000 + 0.00000e+00i, 64.0000e+000 + 0.00000e+00i, 64.0000e+000 + 0.00000e+00i, 64.0000e+000 + 0.00000e+00i,
			64.0000e+000 + 64.0000e+000i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 64.0000e+000 + 64.0000e+000i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 64.0000e+000 + 64.0000e+000i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i,
			0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 0.0000e-003 + 0.00000e+00i, 64.0000e+000 + 64.0000e+000i, 0.0000e-003 + 0.00000e+00i,
		},
		{
			0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i, 0.10000e+01 + 0.10000e+01i,
			0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.10000e+01 + 0.10000e+01i,
		},
		{
			6.4000e+01 + 0.00000e+00i, 2.5000e-01 + 0.00000e+00i, 5.00000e-01 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, -2.0000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 4.0000e+00 + 0.00000e+00i, 2.00000e+00 + 0.00000e+00i, 4.0960e+00 + 0.00000e+00i, 1.6000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 1.0240e+01 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 5.0000e-01 + 0.00000e+00i, 3.00000e+00 + 0.00000e+00i, 4.0960e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, -6.4000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, -3.90625e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, -3.1250e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 8.0000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, -2.0000e+00 + 0.00000e+00i, 4.00000e+00 + 0.00000e+00i, 1.6000e+00 + 0.00000e+00i, 2.0000e+00 + 0.00000e+00i, -8.0000e+00 + 0.00000e+00i, 8.0000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 6.0000e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.00000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i,
		},
		{
			1.0000e+03 + 0.00000e+00i, 3.1250e-02 + 0.00000e+00i, 3.7500e-01 + 0.00000e+00i, 6.2500e-02 + 0.00000e+00i, 3.90625e+03 + 0.00000e+00i,
			5.7600e+02 + 0.00000e+00i, 0.0000e+00 + 0.00000e+00i, 1.6000e-03 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, 1.5000e+00 + 0.00000e+00i,
			0.0000e+00 + 0.00000e+00i, -3.7500e+01 + 0.00000e+00i, 2.0000e+00 + 0.00000e+00i, 1.2500e-01 + 0.00000e+00i, 6.2500e-02 + 0.00000e+00i,
			5.7600e+02 + 0.00000e+00i, 2.0000e-03 + 0.00000e+00i, 8.0000e+00 + 0.00000e+00i, 1.0000e+00 + 0.00000e+00i, -5.0000e+02 + 0.00000e+00i,
			7.6800e+02 + 0.00000e+00i, 4.0000e+02 + 0.00000e+00i, 1.6000e+01 + 0.00000e+00i, 1.2000e+03 + 0.00000e+00i, 3.0000e+00 + 0.00000e+00i,
		},
		{
			1.000000000000000000e+00 + 0.0000e+00i, 6.344854593289122931e+03 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i,
			1.576080247855779135e-04 + 0.0000e+00i, 1.000000000000000000e+00 + 0.0000e+00i, 6.344854593289122931e+03 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i,
			0.000000000000000000e+00 + 0.0000e+00i, 1.576080247855779135e-04 + 0.0000e+00i, 1.000000000000000000e+00 + 0.0000e+00i, 3.172427296644561466e+03 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i,
			0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 3.152160495711558270e-04 + 0.0000e+00i, 1.000000000000000000e+00 + 0.0000e+00i, 1.586213648322280733e+03 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i,
			0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 6.304320991423116539e-04 + 0.0000e+00i, 1.000000000000000000e+00 + 0.0000e+00i, 1.586213648322280733e+03 + 0.0000e+00i,
			0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 0.000000000000000000e+00 + 0.0000e+00i, 6.304320991423116539e-04 + 0.0000e+00i, 1.000000000000000000e+00 + 0.0000e+00i,
		},
	}
	scalinlist := [][]float64{
		{0.10000e+01, 0.20000e+01, 0.30000e+01, 0.40000e+01, 0.50000e+01},
		{0.10000e+01, 0.20000e+01, 0.30000e+01, 0.20000e+01, 0.10000e+01},
		{0.10000e+01, 0.20000e+01, 0.30000e+01, 0.20000e+01, 0.10000e+01},
		{6.25000e-02, 6.25000e-02, 2.00000e+00, 2.00000e+00},
		{0.40000e+01, 0.30000e+01, 0.50000e+01, 0.80000e+01, 0.12500e+00, 0.10000e+01},
		{64.0000e+000, 500.0000e-003, 62.5000e-003, 4.0000e+000, 2.0000e+000},
		{1.0000e+000, 1.0000e+000, 2.0000e+000, 1.0000e+000},
		{1.0000e+000, 1.0000e+000, 1.0000e+000, 500.0000e-003},
		{128.0000e+000, 16.0000e+000, 2.0000e+000, 250.0000e-003, 31.2500e-003},
		{0.30000e+01, 0.10000e+01, 0.10000e+01, 0.10000e+01, 0.10000e+01, 0.40000e+01},
		{3.0000e+00, 1.953125e-03, 3.1250e-02, 3.2000e+01, 2.5000e-01, 1.0000e+00, 6.0000e+00},
		{1.2800e+02, 2.0000e+00, 1.6000e+01, 2.0000e+00, 1.0000e+00},
		{2.494800386918399765e+291, 1.582914569427869018e+175, 1.004336277661868922e+59, 3.186183822264904554e-58, 5.053968264940243633e-175, 8.016673440035891112e-292},
	}

	for _i, n = range nlist {
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}

		iloin = iloinlist[_i]
		ihiin = ihiinlist[_i]
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				ain.Set(i-1, j-1, ainlist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			scalin.Set(i-1, scalinlist[_i][i-1])
		}

		knt = knt + 1
		golapack.Zgebal('B', &n, a, &lda, &ilo, &ihi, scale, &info)

		if info != 0 {
			t.Fail()
			ninfo = ninfo + 1
			lmax[0] = knt
		}

		if ilo != iloin || ihi != ihiin {
			t.Fail()
			ninfo = ninfo + 1
			lmax[1] = knt
		}

		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				temp = maxf64(cabs1(a.Get(i-1, j-1)), cabs1(ain.Get(i-1, j-1)))
				temp = maxf64(temp, sfmin)
				vmax = maxf64(vmax, cabs1(a.Get(i-1, j-1)-ain.Get(i-1, j-1))/temp)
			}
		}

		for i = 1; i <= n; i++ {
			temp = maxf64(scale.Get(i-1), scalin.Get(i-1))
			temp = maxf64(temp, sfmin)
			vmax = maxf64(vmax, math.Abs(scale.Get(i-1)-scalin.Get(i-1))/temp)
		}

		if vmax > rmax {
			lmax[2] = knt
			rmax = vmax
		}

	}

	fmt.Printf(" .. test output of ZGEBAL .. \n")

	fmt.Printf(" value of largest test error            = %12.3E\n", rmax)
	fmt.Printf(" example number where info is not zero  = %4d\n", lmax[0])
	fmt.Printf(" example number where ILO or IHI wrong  = %4d\n", lmax[1])
	fmt.Printf(" example number having largest error    = %4d\n", lmax[2])
	fmt.Printf(" number of examples where info is not 0 = %4d\n", ninfo)
	fmt.Printf(" total number of examples tested        = %4d\n\n", knt)
}
