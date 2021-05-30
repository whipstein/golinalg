package eig

import (
	"fmt"
	"golinalg/golapack"
	"math"
	"testing"
)

// Dchkbl tests DGEBAL, a routine for balancing a general real
// matrix and isolating some of its eigenvalues.
func Dchkbl(t *testing.T) {
	var rmax, sfmin, temp, vmax, zero float64
	var _i, i, ihi, ihiin, ilo, iloin, info, j, knt, lda, n, ninfo int
	_ = _i

	// dummy := vf(1)
	scale := vf(20)
	scalin := vf(20)
	lmax := make([]int, 3)
	a := mf(20, 20, opts)
	ain := mf(20, 20, opts)

	lda = 20
	zero = 0.0

	alenlist := []int{5, 5, 5, 4, 6, 5, 4, 4, 5, 6, 7, 5, 6}
	alist := [][]float64{
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.5000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.0000e+00,
			0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
		},
		{
			0.0000e+00, 0.2000e+01, 0.1000e+00, 0.0000e+00,
			0.2000e+01, 0.0000e+00, 0.0000e+00, 0.1000e+00,
			0.1000e+03, 0.0000e+00, 0.0000e+00, 0.2000e+01,
			0.0000e+00, 0.1000e+03, 0.2000e+01, 0.0000e+00,
		},
		{
			0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1024e+04,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1280e+03,
			0.0000e+00, 0.2000e+01, 0.3000e+04, 0.0000e+00, 0.0000e+00, 0.2000e+01,
			0.1280e+03, 0.4000e+01, 0.4000e-02, 0.5000e+01, 0.6000e+03, 0.8000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e-02, 0.2000e+01,
			0.8000e+01, 0.8192e+04, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01,
		},
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.8000e+01,
			0.0000e+00, 0.2000e+01, 0.8192e+04, 0.2000e+01, 0.4000e+01,
			0.2500e-03, 0.1250e-03, 0.4000e+01, 0.0000e+00, 0.6400e+02,
			0.0000e+00, 0.2000e+01, 0.1024e+04, 0.4000e+01, 0.8000e+01,
			0.0000e+00, 0.8192e+04, 0.0000e+00, 0.0000e+00, 0.8000e+01,
		},
		{
			0.1000e+01, 0.1000e+07, 0.1000e+07, 0.1000e+07,
			-0.2000e+07, 0.3000e+01, 0.2000e-05, 0.3000e-05,
			-0.3000e+07, 0.0000e+00, 0.1000e-05, 0.2000e+01,
			0.1000e+07, 0.0000e+00, 0.3000e-05, 0.4000e+07,
		},
		{
			0.1000e+01, 0.1000e+05, 0.1000e+05, 0.1000e+05,
			-0.2000e+05, 0.3000e+01, 0.2000e-02, 0.3000e-02,
			0.0000e+00, 0.2000e+01, 0.0000e+00, -0.3000e+05,
			0.0000e+00, 0.0000e+00, 0.1000e+05, 0.0000e+00,
		},
		{
			0.1000e+01, 0.5120e+03, 0.4096e+04, 3.2768e+04, 2.62144e+05,
			0.8000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.8000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.8000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.8000e+01, 0.0000e+00,
		},
		{
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01,
		},
		{
			0.6000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.4000e+01, 0.0000e+00, 0.2500e-03, 0.1250e-01, 0.2000e-01, 0.1250e+00,
			0.1000e+01, 0.1280e+03, 0.6400e+02, 0.0000e+00, 0.0000e+00, -0.2000e+01, 0.1600e+02,
			0.0000e+00, 1.6384e+04, 0.0000e+00, 0.1000e+01, -0.4000e+03, 0.2560e+03, -0.4000e+04,
			-0.2000e+01, -0.2560e+03, 0.0000e+00, 0.1250e-01, 0.2000e+01, 0.2000e+01, 0.3200e+02,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.8000e+01, 0.0000e+00, 0.4000e-02, 0.1250e+00, -0.2000e+00, 0.3000e+01,
		},
		{
			0.1000e+04, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+06,
			0.9000e+01, 0.0000e+00, 0.2000e-03, 0.1000e+01, 0.3000e+01,
			0.0000e+00, -0.3000e+03, 0.2000e+01, 0.1000e+01, 0.1000e+01,
			0.9000e+01, 0.2000e-02, 0.1000e+01, 0.1000e+01, -0.1000e+04,
			0.6000e+01, 0.2000e+03, 0.1000e+01, 0.6000e+03, 0.3000e+01,
		},
		{
			1.0000e+00, 1.0000e+120, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			1.0000e-120, 1.0000e+00, 1.0000e+120, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 1.0000e-120, 1.0000e+00, 1.0000e+120, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 1.0000e-120, 1.0000e+00, 1.0000e+120, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e-120, 1.0000e+00, 1.0000e+120,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e-120, 1.0000e+00,
		},
	}
	ilolist := []int{1, 1, 1, 1, 4, 1, 1, 1, 1, 2, 2, 1, 1}
	ihilist := []int{1, 1, 1, 4, 6, 5, 4, 4, 5, 5, 5, 5, 6}
	ainlist := [][]float64{
		{
			0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.2000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.4000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.5000e+01,
		},
		{
			0.5000e+01, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.4000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.3000e+01, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01, 0.0000e+00,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			0.0000e-03, 2.0000e+00, 3.2000e+00, 0.0000e-03,
			2.0000e+00, 0.0000e-03, 0.0000e-03, 3.2000e+00,
			3.1250e+00, 0.0000e-03, 0.0000e-03, 2.0000e+00,
			0.0000e-03, 3.1250e+00, 2.0000e+00, 0.0000e-03,
		},
		{
			0.5000e+01, 0.4000e-02, 0.6000e+03, 0.1024e+04, 0.5000e+00, 0.8000e+01,
			0.0000e+00, 0.3000e+04, 0.0000e+00, 0.0000e+00, 0.2500e+00, 0.2000e+01,
			0.0000e+00, 0.0000e+00, 0.2000e-02, 0.0000e+00, 0.0000e+00, 0.2000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.2000e+01, 0.0000e+00, 0.1280e+03,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1024e+04,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.6400e+02, 0.1024e+04, 0.2000e+01,
		},
		{
			1.0000e+00, 0.0000e-03, 0.0000e-03, 0.0000e-03, 2.0000e+00,
			0.0000e-03, 2.0000e+00, 1.0240e+03, 16.0000e+00, 16.0000e+00,
			3.2000e-02, 1.0000e-03, 4.0000e+00, 0.0000e-03, 2.0480e+03,
			0.0000e-03, 250.0000e-03, 16.0000e+00, 4.0000e+00, 4.0000e+00,
			0.0000e-03, 2.0480e+03, 0.0000e-03, 0.0000e-03, 8.0000e+00,
		},
		{
			1.0000e+00, 1.0000e+06, 2.0000e+06, 1.0000e+06,
			-2.0000e+06, 3.0000e+00, 4.0000e-06, 3.0000e-06,
			-1.5000e+06, 0.0000e-03, 1.0000e-06, 1.0000e+00,
			1.0000e+06, 0.0000e-03, 6.0000e-06, 4.0000e+06,
		},
		{
			1.0000e+00, 10.0000e+03, 10.0000e+03, 5.0000e+03,
			-20.0000e+03, 3.0000e+00, 2.0000e-03, 1.5000e-03,
			0.0000e-03, 2.0000e+00, 0.0000e-03, -15.0000e+03,
			0.0000e-03, 0.0000e-03, 20.0000e+03, 0.0000e-03,
		},
		{
			1.0000e+00, 32.0000e+00, 32.0000e+00, 32.0000e+000, 32.0000e+00,
			128.0000e+00, 0.0000e-03, 0.0000e-03, 0.0000e-003, 0.0000e-03,
			0.0000e-03, 64.0000e+00, 0.0000e-03, 0.0000e-003, 0.0000e-03,
			0.0000e-03, 0.0000e-03, 64.0000e+00, 0.0000e-003, 0.0000e-03,
			0.0000e-03, 0.0000e-03, 0.0000e-03, 64.0000e+000, 0.0000e-03,
		},
		{
			0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01,
			0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.1000e+01,
		},
		{
			6.4000e+01, 1.0000e+00, 5.00000e-01, 0.0000e+00, 0.0000e+00, 1.0000e+00, -2.0000e+00,
			0.0000e+00, 4.0000e+00, 5.00000e-01, 1.0240e+00, 8.0000e-01, 0.0000e+00, 2.5600e+00,
			0.0000e+00, 2.0000e+00, 3.00000e+00, 4.0960e+00, 2.0000e+00, 0.0000e+00, -6.4000e+00,
			0.0000e+00, 4.0000e+00, -3.90625e+00, 1.0000e+00, -6.2500e+00, 0.0000e+00, 8.0000e+00,
			0.0000e+00, -4.0000e+00, 2.00000e+00, 8.0000e-01, 2.0000e+00, -4.0000e+00, 4.0000e+00,
			0.0000e+00, 0.0000e+00, 0.00000e+00, 0.0000e+00, 0.0000e+00, 6.0000e+00, 1.0000e+00,
			0.0000e+00, 0.0000e+00, 0.00000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
		},
		{
			1.0000e+03, 3.1250e-02, 3.7500e-01, 3.1250e-02, 1.95312500e+03,
			5.7600e+02, 0.0000e+00, 1.6000e-03, 5.0000e-01, 7.50000000e-01,
			0.0000e+00, -3.7500e+01, 2.0000e+00, 6.2500e-02, 3.12500000e-02,
			1.1520e+03, 4.0000e-03, 1.6000e+01, 1.0000e+00, -5.00000000e+02,
			1.5360e+03, 8.0000e+02, 3.2000e+01, 1.2000e+03, 3.00000000e+00,
		},
		{
			0.10000000000000000000e+01, 0.63448545932891229313e+04, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00,
			0.15760802478557791348e-03, 0.10000000000000000000e+01, 0.63448545932891229313e+04, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00,
			0.00000000000000000000e+00, 0.15760802478557791348e-03, 0.10000000000000000000e+01, 0.31724272966445614657e+04, 0.00000000000000000000e+00, 0.00000000000000000000e+00,
			0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.31521604957115582695e-03, 0.10000000000000000000e+01, 0.15862136483222807328e+04, 0.00000000000000000000e+00,
			0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.63043209914231165391e-03, 0.10000000000000000000e+01, 0.79310682416114036641e+03,
			0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.00000000000000000000e+00, 0.12608641982846233078e-02, 0.10000000000000000000e+01,
		},
	}
	scalelist := [][]float64{
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.4000e+01, 0.5000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{0.1000e+01, 0.2000e+01, 0.3000e+01, 0.2000e+01, 0.1000e+01},
		{62.5000e-03, 62.5000e-03, 2.0000e+00, 2.0000e+00},
		{0.4000e+01, 0.3000e+01, 0.5000e+01, 0.8000e+01, 0.1250e+00, 0.1000e+01},
		{8.0000e+00, 500.0000e-03, 62.5000e-03, 4.0000e+00, 2.0000e+00},
		{1.0000e+00, 1.0000e+00, 2.0000e+00, 1.0000e+00},
		{1.0000e+00, 1.0000e+00, 1.0000e+00, 500.0000e-03},
		{256.0000e+00, 16.0000e+00, 2.0000e+00, 250.0000e-03, 31.2500e-03},
		{0.3000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.1000e+01, 0.4000e+01},
		{3.0000e+00, 7.812500e-03, 3.1250e-02, 3.2000e+01, 5.0000e-01, 1.0000e+00, 6.0000e+00},
		{3.2000e+01, 5.0000e-01, 4.0000e+00, 2.5000e-01, 1.2500e-01},
		{2.494800386918399765e+291, 1.582914569427869018e+175, 1.004336277661868922e+59, 3.186183822264904554e-58, 5.053968264940243633e-175, 0.40083367200179455560e-291},
	}

	lmax[0] = 0
	lmax[1] = 0
	lmax[2] = 0
	ninfo = 0
	knt = 0
	rmax = zero
	vmax = zero
	sfmin = golapack.Dlamch(SafeMinimum)
	// meps = golapack.Dlamch(Epsilon)

	for _i, n = range alenlist {
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				a.Set(i-1, j-1, alist[_i][(i-1)*(n)+j-1])
			}
		}

		iloin = ilolist[_i]
		ihiin = ihilist[_i]
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				ain.Set(i-1, j-1, ainlist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			scalin.Set(i-1, scalelist[_i][i-1])
		}

		// anorm = Dlange('M', &n, &n, a, &lda, dummy)
		knt = knt + 1

		golapack.Dgebal('B', &n, a, &lda, &ilo, &ihi, scale, &info)

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
				temp = maxf64(a.Get(i-1, j-1), ain.Get(i-1, j-1))
				temp = maxf64(temp, sfmin)
				vmax = maxf64(vmax, math.Abs(a.Get(i-1, j-1)-ain.Get(i-1, j-1))/temp)
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

	fmt.Printf(" .. test output of DGEBAL .. \n")

	fmt.Printf(" value of largest test error            = %12.3E\n", rmax)
	fmt.Printf(" example number where info is not zero  = %4d\n", lmax[0])
	fmt.Printf(" example number where ILO or IHI wrong  = %4d\n", lmax[1])
	fmt.Printf(" example number having largest error    = %4d\n", lmax[2])
	fmt.Printf(" number of examples where info is not 0 = %4d\n", ninfo)
	fmt.Printf(" total number of examples tested        = %4d\n", knt)
}
