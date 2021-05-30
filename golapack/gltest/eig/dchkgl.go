package eig

import (
	"fmt"
	"golinalg/golapack"
	"math"
	"testing"
)

// Dchkgl tests DGGBAL, a routine for balancing a matrix pair (A, B).
func Dchkgl(t *testing.T) {
	var anorm, bnorm, eps, rmax, vmax, zero float64
	var _i, i, ihi, ihiin, ilo, iloin, info, j, knt, lda, ldb, lwork, n, ninfo int

	lda = 20
	ldb = 20
	lwork = 6 * lda
	zero = 0.0
	lscale := vf(20)
	lsclin := vf(20)
	rscale := vf(20)
	rsclin := vf(20)
	work := vf(lwork)
	lmax := make([]int, 5)
	a := mf(20, 20, opts)
	ain := mf(20, 20, opts)
	b := mf(20, 20, opts)
	bin := mf(20, 20, opts)

	nlist := []int{6, 6, 6, 5, 6, 6, 7, 6}
	alist := [][]float64{
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.5000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.6000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.6000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01,
		},
		{
			0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-05, 0.1000e+07,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-05, 0.1000e-05,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+07, 0.1000e+07,
		},
		{
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
		},
		{
			-0.2000e+02, -0.1000e+05, -0.2000e+01, -0.1000e+07, -0.1000e+02, -0.2000e+06,
			0.6000e-02, 0.4000e+01, 0.6000e-03, 0.2000e+03, 0.3000e-02, 0.3000e+02,
			-0.2000e+00, -0.3000e+03, -0.4000e-01, -0.1000e+05, 0.0000e+00, 0.3000e+04,
			0.6000e-04, 0.4000e-01, 0.9000e-05, 0.9000e+01, 0.3000e-04, 0.5000e+00,
			0.6000e-01, 0.5000e+02, 0.8000e-02, -0.4000e+04, 0.8000e-01, 0.0000e+00,
			0.0000e+00, 0.1000e+04, 0.7000e+00, -0.2000e+06, 0.1300e+02, -0.6000e+05,
		},
	}
	blist := [][]float64{
		{
			0.6000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.6000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+11,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-05, 0.1000e+07,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-05, 0.1000e-05,
			0.1000e+07, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+07, 0.1000e+07,
		},
		{
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
		},
		{
			-0.2000e+02, -0.1000e+05, 0.2000e+01, -0.2000e+07, 0.1000e+02, -0.1000e+06,
			0.5000e-02, 0.3000e+01, -0.2000e-03, 0.4000e+03, -0.1000e-02, 0.3000e+02,
			0.0000e+00, -0.1000e+03, -0.8000e-01, 0.2000e+05, -0.4000e+00, 0.0000e+00,
			0.5000e-04, 0.3000e-01, 0.2000e-05, 0.4000e+01, 0.2000e-04, 0.1000e+00,
			0.4000e-01, 0.3000e+02, -0.1000e-02, 0.3000e+04, -0.1000e-01, 0.6000e+03,
			-0.1000e+01, 0.0000e+00, 0.4000e+00, -0.1000e+06, 0.4000e+01, 0.2000e+05,
		},
	}
	iloinlist := []int{1, 1, 1, 1, 1, 4, 3, 1}
	ihiinlist := []int{1, 1, 1, 1, 6, 6, 5, 6}
	ainlist := [][]float64{
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.5000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.6000e+01,
		},
		{
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.6000e+01, 0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e-03, 0.1000e+05, 0.1000e+04, 0.1000e+02, 0.1000e+00, 0.1000e-01,
			0.1000e-02, 0.1000e-04, 0.1000e+05, 0.1000e+03, 0.1000e+01, 0.1000e+00,
			0.1000e+00, 0.1000e-02, 0.1000e-03, 0.1000e+05, 0.1000e+03, 0.1000e+02,
			0.1000e+02, 0.1000e+00, 0.1000e-01, 0.1000e-03, 0.1000e+05, 0.1000e+04,
			0.1000e+03, 0.1000e+01, 0.1000e+00, 0.1000e-02, 0.1000e-04, 0.1000e+05,
			0.1000e+05, 0.1000e+03, 0.1000e+02, 0.1000e+00, 0.1000e-02, 0.1000e-03,
		},
		{
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e-03, 0.1000e+05,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+05, 0.1000e+01, 0.1000e-03,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-03, 0.1000e+05, 0.1000e+01,
		},
		{
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			-0.2000e+00, -0.1000e+01, -0.2000e+00, -0.1000e+01, -0.1000e+01, -0.2000e+01,
			0.6000e+00, 0.4000e+01, 0.6000e+00, 0.2000e+01, 0.3000e+01, 0.3000e+01,
			-0.2000e+00, -0.3000e+01, -0.4000e+00, -0.1000e+01, 0.0000e+00, 0.3000e+01,
			0.6000e+00, 0.4000e+01, 0.9000e+00, 0.9000e+01, 0.3000e+01, 0.5000e+01,
			0.6000e+00, 0.5000e+01, 0.8000e+00, -0.4000e+01, 0.8000e+01, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.7000e+00, -0.2000e+01, 0.1300e+02, -0.6000e+01,
		},
	}
	binlist := [][]float64{
		{
			0.6000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.5000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.6000e+01, 0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e-03, 0.1000e+05, 0.1000e+04, 0.1000e+02, 0.1000e+00, 0.1000e-01,
			0.1000e-02, 0.1000e-04, 0.1000e+05, 0.1000e+03, 0.1000e+01, 0.1000e+00,
			0.1000e+00, 0.1000e-02, 0.1000e-03, 0.1000e+05, 0.1000e+03, 0.1000e+02,
			0.1000e+02, 0.1000e+00, 0.1000e-01, 0.1000e-03, 0.1000e+05, 0.1000e+04,
			0.1000e+03, 0.1000e+01, 0.1000e+00, 0.1000e-02, 0.1000e-04, 0.1000e+05,
			0.1000e+05, 0.1000e+03, 0.1000e+02, 0.1000e+00, 0.1000e-02, 0.1000e-03,
		},
		{
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e-03, 0.1000e+05,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+05, 0.1000e+01, 0.1000e-03,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e-03, 0.1000e+05, 0.1000e+01,
		},
		{
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			-0.2000e+00, -0.1000e+01, 0.2000e+00, -0.2000e+01, 0.1000e+01, -0.1000e+01,
			0.5000e+00, 0.3000e+01, -0.2000e+00, 0.4000e+01, -0.1000e+01, 0.3000e+01,
			0.0000e+00, -0.1000e+01, -0.8000e+00, 0.2000e+01, -0.4000e+01, 0.0000e+00,
			0.5000e+00, 0.3000e+01, 0.2000e+00, 0.4000e+01, 0.2000e+01, 0.1000e+01,
			0.4000e+00, 0.3000e+01, -0.1000e+00, 0.3000e+01, -0.1000e+01, 0.6000e+01,
			-0.1000e+00, 0.0000e+00, 0.4000e+00, -0.1000e+01, 0.4000e+01, 0.2000e+01,
		},
	}
	lsclinlist := [][]float64{
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.6000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e-05, 0.1000e-04, 0.1000e-02, 0.1000e+00, 0.1000e+01, 0.1000e+03},
		{0.4000e+01, 0.4000e+01, 0.4000e+01, 0.1000e+00, 0.1000e+04, 0.1000e-04},
		{0.3000e+01, 0.2000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.6000e+01, 0.5000e+01},
		{0.1000e-02, 0.1000e+02, 0.1000e+00, 0.1000e+04, 0.1000e+01, 0.1000e-01},
	}
	rsclinlist := [][]float64{
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01, 0.6000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+03, 0.1000e+01, 0.1000e+00, 0.1000e-02, 0.1000e-04, 0.1000e-05},
		{0.2000e+01, 0.3000e+01, 0.4000e+01, 0.1000e-04, 0.1000e+04, 0.1000e+00},
		{0.1000e+01, 0.3000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.2000e+01, 0.2000e+01},
		{0.1000e+02, 0.1000e+00, 0.1000e+03, 0.1000e-02, 0.1000e+03, 0.1000e-01},
	}

	lmax[0] = 0
	lmax[1] = 0
	lmax[2] = 0
	ninfo = 0
	knt = 0
	rmax = zero

	eps = golapack.Dlamch(Precision)

	for _i, n = range nlist {
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}

		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				b.Set(i-1, j-1, blist[_i][(i-1)*(n)+j-1])
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
			for j = 1; j <= n; j++ {
				bin.Set(i-1, j-1, binlist[_i][(i-1)*(n)+j-1])
			}
		}

		for i = 1; i <= n; i++ {
			lsclin.Set(i-1, lsclinlist[_i][i-1])
		}
		for i = 1; i <= n; i++ {
			rsclin.Set(i-1, rsclinlist[_i][i-1])
		}

		anorm = golapack.Dlange('M', &n, &n, a, &lda, work)
		bnorm = golapack.Dlange('M', &n, &n, b, &ldb, work)

		knt = knt + 1

		golapack.Dggbal('B', &n, a, &lda, b, &ldb, &ilo, &ihi, lscale, rscale, work, &info)

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

		vmax = zero
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				vmax = maxf64(vmax, math.Abs(a.Get(i-1, j-1)-ain.Get(i-1, j-1)))
				vmax = maxf64(vmax, math.Abs(b.Get(i-1, j-1)-bin.Get(i-1, j-1)))
			}
		}

		for i = 1; i <= n; i++ {
			vmax = maxf64(vmax, math.Abs(lscale.Get(i-1)-lsclin.Get(i-1)))
			vmax = maxf64(vmax, math.Abs(rscale.Get(i-1)-rsclin.Get(i-1)))
		}

		vmax = vmax / (eps * maxf64(anorm, bnorm))

		if vmax > rmax {
			lmax[2] = knt
			rmax = vmax
		}

	}

	fmt.Printf(" .. test output of DGGBAL .. \n")

	fmt.Printf(" value of largest test error            = %12.3E\n", rmax)
	fmt.Printf(" example number where info is not zero  = %4d\n", lmax[0])
	fmt.Printf(" example number where ILO or IHI wrong  = %4d\n", lmax[1])
	fmt.Printf(" example number having largest error    = %4d\n", lmax[2])
	fmt.Printf(" number of examples where info is not 0 = %4d\n", ninfo)
	fmt.Printf(" total number of examples tested        = %4d\n", knt)
}