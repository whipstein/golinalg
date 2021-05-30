package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dget38 tests DTRSEN, a routine for estimating condition numbers of a
// cluster of eigenvalues and/or its associated right invariant subspace
//
// The test matrices are read from a file with logical unit number NIN.
func Dget38(rmax *mat.Vector, lmax, ninfo *[]int, knt *int) {
	var bignum, eps, epsin, one, s, sep, sepin, septmp, sin, smlnum, stmp, tnrm, tol, tolin, two, v, vimin, vmax, vmul, vrmin, zero float64
	var _i, i, info, iscl, itmp, j, kmin, ldt, liwork, lwork, m, n, ndim int

	zero = 0.0
	one = 1.0
	two = 2.0
	epsin = 5.9605e-8
	ldt = 20
	lwork = 2 * ldt * (10 + ldt)
	liwork = ldt * ldt
	_select := make([]bool, 20)
	result := vf(2)
	val := vf(3)
	wi := vf(20)
	witmp := vf(20)
	work := vf(lwork)
	wr := vf(20)
	wrtmp := vf(20)
	ipnt := make([]int, 20)
	iselec := make([]int, 20)
	iwork := make([]int, liwork)
	q := mf(20, 20, opts)
	qsav := mf(20, 20, opts)
	qtmp := mf(20, 20, opts)
	t := mf(20, 20, opts)
	tmp := mf(20, 20, opts)
	tsav := mf(20, 20, opts)
	tsav1 := mf(20, 20, opts)
	ttmp := mf(20, 20, opts)

	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     EPSIN = 2**(-24) = precision to which input data computed
	eps = maxf64(eps, epsin)
	rmax.Set(0, zero)
	rmax.Set(1, zero)
	rmax.Set(2, zero)
	(*lmax)[0] = 0
	(*lmax)[1] = 0
	(*lmax)[2] = 0
	(*knt) = 0
	(*ninfo)[0] = 0
	(*ninfo)[1] = 0
	(*ninfo)[2] = 0

	val.Set(0, math.Sqrt(smlnum))
	val.Set(1, one)
	val.Set(2, math.Sqrt(math.Sqrt(bignum)))

	nlist := []int{1, 1, 6, 6, 6, 6, 6, 2, 4, 7, 4, 7, 6, 8, 9, 10, 12, 12, 3, 5, 6, 6, 6, 5, 6, 10}
	ndimlist := []int{1, 1, 3, 3, 3, 3, 3, 1, 2, 6, 2, 5, 4, 4, 3, 4, 6, 7, 1, 1, 4, 2, 3, 1, 2, 1}
	iseleclist := [][]int{
		{1},
		{1},
		{4, 5, 6},
		{4, 5, 6},
		{4, 5, 6},
		{4, 5, 6},
		{4, 5, 6},
		{1},
		{1, 2},
		{1, 2, 3, 4, 5, 6},
		{2, 3},
		{1, 2, 3, 4, 5},
		{3, 4, 5, 6},
		{1, 2, 3, 4},
		{1, 2, 3},
		{1, 2, 3, 4},
		{1, 2, 3, 4, 5, 6},
		{6, 7, 8, 9, 10, 11, 12},
		{1},
		{3},
		{1, 2, 3, 5},
		{3, 4},
		{3, 4, 5},
		{1},
		{1, 2},
		{1},
	}
	tmplist := [][]float64{
		{
			0.00000e+00,
		},
		{
			1.00000e+00,
		},
		{
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 2.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 3.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 4.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.00000e+00,
		},
		{
			1.00000e+00, 2.00000e+00,
			0.00000e+00, 3.00000e+00,
		},
		{
			8.52400e-01, 5.61100e-01, 7.04300e-01, 9.54000e-01,
			2.79800e-01, 7.21600e-01, 9.61300e-01, 3.58200e-01,
			7.08100e-01, 4.09400e-01, 2.25000e-01, 9.51800e-01,
			5.54300e-01, 5.22000e-01, 6.86000e-01, 3.07000e-02,
		},
		{
			7.81800e-01, 5.65700e-01, 7.62100e-01, 7.43600e-01, 2.55300e-01, 4.10000e-01, 1.34000e-02,
			6.45800e-01, 2.66600e-01, 5.51000e-01, 8.31800e-01, 9.27100e-01, 6.20900e-01, 7.83900e-01,
			1.31600e-01, 4.91400e-01, 1.77100e-01, 1.96400e-01, 1.08500e-01, 9.27000e-01, 2.24700e-01,
			6.41000e-01, 4.68900e-01, 9.65900e-01, 8.88400e-01, 3.76900e-01, 9.67300e-01, 6.18300e-01,
			8.38200e-01, 8.74300e-01, 4.50700e-01, 9.44200e-01, 7.75500e-01, 9.67600e-01, 7.83100e-01,
			3.25900e-01, 7.38900e-01, 8.30200e-01, 4.52100e-01, 3.01500e-01, 2.13300e-01, 8.43400e-01,
			5.24400e-01, 5.01600e-01, 7.52900e-01, 3.83800e-01, 8.47900e-01, 9.12800e-01, 5.77000e-01,
		},
		{
			-9.85900e-01, 1.47840e+00, -1.33600e-01, -2.95970e+00,
			-4.33700e-01, -6.54000e-01, -7.15500e-01, 1.23760e+00,
			-7.36300e-01, -1.97680e+00, -1.95100e-01, 3.43200e-01,
			6.41400e-01, -1.40880e+00, 6.39400e-01, 8.58000e-02,
		},
		{
			2.72840e+00, 2.15200e-01, -1.05200e+00, -2.44600e-01, -6.53000e-02, 3.90500e-01, 1.40980e+00,
			9.75300e-01, 6.51500e-01, -4.76200e-01, 5.42100e-01, 6.20900e-01, 4.75900e-01, -1.44930e+00,
			-9.05200e-01, 1.79000e-01, -7.08600e-01, 4.62100e-01, 1.05800e+00, 2.24260e+00, 1.58260e+00,
			-7.17900e-01, -2.53400e-01, -4.73900e-01, -1.08100e+00, 4.13800e-01, -9.50000e-02, 1.45300e-01,
			-1.37990e+00, -1.06490e+00, 1.25580e+00, 7.80100e-01, -6.40500e-01, -8.61000e-02, 8.30000e-02,
			2.84900e-01, -1.29900e-01, 4.80000e-02, -2.58600e-01, 4.18900e-01, 1.37680e+00, 8.20800e-01,
			-5.44200e-01, 9.74900e-01, 9.55800e-01, 1.23700e-01, 1.09020e+00, -1.40600e-01, 1.90960e+00,
		},
		{
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			1.00000e-06, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 1.00000e+01, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 1.00000e+01, 1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 1.00000e+01,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.50000e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.50000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.50000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.75000e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.75000e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.75000e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.75000e-01,
		},
		{
			1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, -1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+01,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.37500e-01,
		},
		{
			1.20000e+01, 1.10000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			1.10000e+01, 1.10000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 1.00000e+01, 1.00000e+01, 9.00000e+00, 8.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 9.00000e+00, 9.00000e+00, 8.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 8.00000e+00, 8.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 7.00000e+00, 7.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 6.00000e+00, 6.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 5.00000e+00, 5.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.00000e+00, 4.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 3.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00000e+00, 2.00000e+00, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			2.00000e-06, 1.00000e+00, -2.00000e+00,
			1.00000e-06, -2.00000e+00, 4.00000e+00,
			0.00000e+00, 1.00000e+00, -2.00000e+00,
		},
		{
			2.00000e-03, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e-03, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, -1.00000e-03, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, -2.00000e-03, 1.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 0.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
		},
		{
			0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00,
			1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
			0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00, 1.00000e+00,
			-1.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00,
			5.00000e-01, 3.33300e-01, 2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01,
			3.33300e-01, 2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01,
			2.50000e-01, 2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01,
			2.00000e-01, 1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01, 1.00000e-01,
			1.66700e-01, 1.42900e-01, 1.25000e-01, 1.11100e-01, 1.00000e-01, 9.09000e-02,
		},
		{
			1.50000e+01, 1.10000e+01, 6.00000e+00, -9.00000e+00, -1.50000e+01,
			1.00000e+00, 3.00000e+00, 9.00000e+00, -3.00000e+00, -8.00000e+00,
			7.00000e+00, 6.00000e+00, 6.00000e+00, -3.00000e+00, -1.10000e+01,
			7.00000e+00, 7.00000e+00, 5.00000e+00, -3.00000e+00, -1.10000e+01,
			1.70000e+01, 1.20000e+01, 5.00000e+00, -1.00000e+01, -1.60000e+01,
		},
		{
			-9.00000e+00, 2.10000e+01, -1.50000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-1.00000e+01, 2.10000e+01, -1.40000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-8.00000e+00, 1.60000e+01, -1.10000e+01, 4.00000e+00, 2.00000e+00, 0.00000e+00,
			-6.00000e+00, 1.20000e+01, -9.00000e+00, 3.00000e+00, 3.00000e+00, 0.00000e+00,
			-4.00000e+00, 8.00000e+00, -6.00000e+00, 0.00000e+00, 5.00000e+00, 0.00000e+00,
			-2.00000e+00, 4.00000e+00, -3.00000e+00, 0.00000e+00, 1.00000e+00, 3.00000e+00,
		},
		{
			1.00000e+00, 1.00000e+00, 1.00000e+00, -2.00000e+00, 1.00000e+00, -1.00000e+00, 2.00000e+00, -2.00000e+00, 4.00000e+00, -3.00000e+00,
			-1.00000e+00, 2.00000e+00, 3.00000e+00, -4.00000e+00, 2.00000e+00, -2.00000e+00, 4.00000e+00, -4.00000e+00, 8.00000e+00, -6.00000e+00,
			-1.00000e+00, 0.00000e+00, 5.00000e+00, -5.00000e+00, 3.00000e+00, -3.00000e+00, 6.00000e+00, -6.00000e+00, 1.20000e+01, -9.00000e+00,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -4.00000e+00, 4.00000e+00, -4.00000e+00, 8.00000e+00, -8.00000e+00, 1.60000e+01, -1.20000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 5.00000e+00, -4.00000e+00, 1.00000e+01, -1.00000e+01, 2.00000e+01, -1.50000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -2.00000e+00, 1.20000e+01, -1.20000e+01, 2.40000e+01, -1.80000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00, 1.50000e+01, -1.30000e+01, 2.80000e+01, -2.10000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00, 1.20000e+01, -1.10000e+01, 3.20000e+01, -2.40000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00, 1.20000e+01, -1.40000e+01, 3.70000e+01, -2.60000e+01,
			-1.00000e+00, 0.00000e+00, 3.00000e+00, -6.00000e+00, 2.00000e+00, -5.00000e+00, 1.20000e+01, -1.40000e+01, 3.60000e+01, -2.50000e+01,
		},
	}
	sinlist := []float64{1.00000e+00, 1.00000e+00, 1.00000e+00, 1.00000e+00, 4.01235e-36, 4.01235e-36, 1.00000e+00, 7.07107e-01, 7.22196e-01, 9.43220e-01, 5.22869e-01, 6.04729e-01, 4.89525e-05, 9.56158e-05, 1.00000e+00, 1.00000e+00, 1.85655e-10, 6.92558e-05, 7.30297e-01, 3.99999e-12, 2.93294e-01, 3.97360e-01, 7.28934e-01, 2.17680e-01, 6.78904e-02, 3.60372e-02}
	sepinlist := []float64{0.00000e+00, 1.00000e+00, 4.43734e-31, 1.19209e-07, 3.20988e-36, 3.20988e-36, 1.00000e+00, 2.00000e+00, 4.63943e-01, 3.20530e+00, 5.45530e-01, 9.00391e-01, 4.56492e-05, 4.14317e-05, 5.55801e-07, 1.16972e-10, 2.20147e-16, 5.52606e-05, 4.00000e+00, 3.99201e-12, 1.63448e-01, 3.58295e-01, 1.24624e-02, 5.22626e-01, 4.22005e-02, 7.96134e-02}

	//     Read input data until N=0.  Assume input eigenvalues are sorted
	//     lexicographically (increasing by real part, then decreasing by
	//     imaginary part)
	for _i, n = range nlist {
		ndim = ndimlist[_i]
		for i = 1; i <= ndim; i++ {
			iselec[i-1] = iseleclist[_i][i-1]
		}
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				tmp.Set(i-1, j-1, tmplist[_i][(i-1)*(n)+j-1])
			}
		}
		sin = sinlist[_i]
		sepin = sepinlist[_i]

		tnrm = golapack.Dlange('M', &n, &n, tmp, &ldt, work)
		for iscl = 1; iscl <= 3; iscl++ {
			//        Scale input matrix
			(*knt) = (*knt) + 1
			golapack.Dlacpy('F', &n, &n, tmp, &ldt, t, &ldt)
			vmul = val.Get(iscl - 1)
			for i = 1; i <= n; i++ {
				goblas.Dscal(&n, &vmul, t.Vector(0, i-1), func() *int { y := 1; return &y }())
			}
			if tnrm == zero {
				vmul = one
			}
			golapack.Dlacpy('F', &n, &n, t, &ldt, tsav, &ldt)

			//        Compute Schur form
			golapack.Dgehrd(&n, func() *int { y := 1; return &y }(), &n, t, &ldt, work.Off(0), work.Off(n+1-1), toPtr(lwork-n), &info)
			if info != 0 {
				(*lmax)[0] = (*knt)
				(*ninfo)[0] = (*ninfo)[0] + 1
				goto label160
			}

			//        Generate orthogonal matrix
			golapack.Dlacpy('L', &n, &n, t, &ldt, q, &ldt)
			golapack.Dorghr(&n, func() *int { y := 1; return &y }(), &n, q, &ldt, work.Off(0), work.Off(n+1-1), toPtr(lwork-n), &info)

			//        Compute Schur form
			golapack.Dhseqr('S', 'V', &n, func() *int { y := 1; return &y }(), &n, t, &ldt, wr, wi, q, &ldt, work, &lwork, &info)
			if info != 0 {
				(*lmax)[1] = (*knt)
				(*ninfo)[1] = (*ninfo)[1] + 1
				goto label160
			}

			//        Sort, _select eigenvalues
			for i = 1; i <= n; i++ {
				ipnt[i-1] = i
				_select[i-1] = false
			}
			goblas.Dcopy(&n, wr, func() *int { y := 1; return &y }(), wrtmp, func() *int { y := 1; return &y }())
			goblas.Dcopy(&n, wi, func() *int { y := 1; return &y }(), witmp, func() *int { y := 1; return &y }())
			for i = 1; i <= n-1; i++ {
				kmin = i
				vrmin = wrtmp.Get(i - 1)
				vimin = witmp.Get(i - 1)
				for j = i + 1; j <= n; j++ {
					if wrtmp.Get(j-1) < vrmin {
						kmin = j
						vrmin = wrtmp.Get(j - 1)
						vimin = witmp.Get(j - 1)
					}
				}
				wrtmp.Set(kmin-1, wrtmp.Get(i-1))
				witmp.Set(kmin-1, witmp.Get(i-1))
				wrtmp.Set(i-1, vrmin)
				witmp.Set(i-1, vimin)
				itmp = ipnt[i-1]
				ipnt[i-1] = ipnt[kmin-1]
				ipnt[kmin-1] = itmp
			}
			for i = 1; i <= ndim; i++ {
				_select[ipnt[iselec[i-1]-1]-1] = true
			}

			//        Compute condition numbers
			golapack.Dlacpy('F', &n, &n, q, &ldt, qsav, &ldt)
			golapack.Dlacpy('F', &n, &n, t, &ldt, tsav1, &ldt)
			golapack.Dtrsen('B', 'V', _select, &n, t, &ldt, q, &ldt, wrtmp, witmp, &m, &s, &sep, work, &lwork, &iwork, &liwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label160
			}
			septmp = sep / vmul
			stmp = s

			//        Compute residuals
			Dhst01(&n, func() *int { y := 1; return &y }(), &n, tsav, &ldt, t, &ldt, q, &ldt, work, &lwork, result)
			vmax = maxf64(result.Get(0), result.Get(1))
			if vmax > rmax.Get(0) {
				rmax.Set(0, vmax)
				if (*ninfo)[0] == 0 {
					(*lmax)[0] = (*knt)
				}
			}

			//        Compare condition number for eigenvalue cluster
			//        taking its condition number into account
			v = maxf64(two*float64(n)*eps*tnrm, smlnum)
			if tnrm == zero {
				v = one
			}
			if v > septmp {
				tol = one
			} else {
				tol = v / septmp
			}
			if v > sepin {
				tolin = one
			} else {
				tolin = v / sepin
			}
			tol = maxf64(tol, smlnum/eps)
			tolin = maxf64(tolin, smlnum/eps)
			if eps*(sin-tolin) > stmp+tol {
				vmax = one / eps
			} else if sin-tolin > stmp+tol {
				vmax = (sin - tolin) / (stmp + tol)
			} else if sin+tolin < eps*(stmp-tol) {
				vmax = one / eps
			} else if sin+tolin < stmp-tol {
				vmax = (stmp - tol) / (sin + tolin)
			} else {
				vmax = one
			}
			if vmax > rmax.Get(1) {
				rmax.Set(1, vmax)
				if (*ninfo)[1] == 0 {
					(*lmax)[1] = (*knt)
				}
			}

			//        Compare condition numbers for invariant subspace
			//        taking its condition number into account
			if v > septmp*stmp {
				tol = septmp
			} else {
				tol = v / stmp
			}
			if v > sepin*sin {
				tolin = sepin
			} else {
				tolin = v / sin
			}
			tol = maxf64(tol, smlnum/eps)
			tolin = maxf64(tolin, smlnum/eps)
			if eps*(sepin-tolin) > septmp+tol {
				vmax = one / eps
			} else if sepin-tolin > septmp+tol {
				vmax = (sepin - tolin) / (septmp + tol)
			} else if sepin+tolin < eps*(septmp-tol) {
				vmax = one / eps
			} else if sepin+tolin < septmp-tol {
				vmax = (septmp - tol) / (sepin + tolin)
			} else {
				vmax = one
			}
			if vmax > rmax.Get(1) {
				rmax.Set(1, vmax)
				if (*ninfo)[1] == 0 {
					(*lmax)[1] = (*knt)
				}
			}

			//        Compare condition number for eigenvalue cluster
			//        without taking its condition number into account
			if sin <= float64(2*n)*eps && stmp <= float64(2*n)*eps {
				vmax = one
			} else if eps*sin > stmp {
				vmax = one / eps
			} else if sin > stmp {
				vmax = sin / stmp
			} else if sin < eps*stmp {
				vmax = one / eps
			} else if sin < stmp {
				vmax = stmp / sin
			} else {
				vmax = one
			}
			if vmax > rmax.Get(2) {
				rmax.Set(2, vmax)
				if (*ninfo)[2] == 0 {
					(*lmax)[2] = (*knt)
				}
			}

			//        Compare condition numbers for invariant subspace
			//        without taking its condition number into account
			if sepin <= v && septmp <= v {
				vmax = one
			} else if eps*sepin > septmp {
				vmax = one / eps
			} else if sepin > septmp {
				vmax = sepin / septmp
			} else if sepin < eps*septmp {
				vmax = one / eps
			} else if sepin < septmp {
				vmax = septmp / sepin
			} else {
				vmax = one
			}
			if vmax > rmax.Get(2) {
				rmax.Set(2, vmax)
				if (*ninfo)[2] == 0 {
					(*lmax)[2] = (*knt)
				}
			}

			//        Compute eigenvalue condition number only and compare
			//        Update Q
			vmax = zero
			golapack.Dlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Dlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Dtrsen('E', 'V', _select, &n, ttmp, &ldt, qtmp, &ldt, wrtmp, witmp, &m, &stmp, &septmp, work, &lwork, &iwork, &liwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label160
			}
			if s != stmp {
				vmax = one / eps
			}
			if -one != septmp {
				vmax = one / eps
			}
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					if ttmp.Get(i-1, j-1) != t.Get(i-1, j-1) {
						vmax = one / eps
					}
					if qtmp.Get(i-1, j-1) != q.Get(i-1, j-1) {
						vmax = one / eps
					}
				}
			}

			//        Compute invariant subspace condition number only and compare
			//        Update Q
			golapack.Dlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Dlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Dtrsen('V', 'V', _select, &n, ttmp, &ldt, qtmp, &ldt, wrtmp, witmp, &m, &stmp, &septmp, work, &lwork, &iwork, &liwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label160
			}
			if -one != stmp {
				vmax = one / eps
			}
			if sep != septmp {
				vmax = one / eps
			}
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					if ttmp.Get(i-1, j-1) != t.Get(i-1, j-1) {
						vmax = one / eps
					}
					if qtmp.Get(i-1, j-1) != q.Get(i-1, j-1) {
						vmax = one / eps
					}
				}
			}

			//        Compute eigenvalue condition number only and compare
			//        Do not update Q
			golapack.Dlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Dlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Dtrsen('E', 'N', _select, &n, ttmp, &ldt, qtmp, &ldt, wrtmp, witmp, &m, &stmp, &septmp, work, &lwork, &iwork, &liwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label160
			}
			if s != stmp {
				vmax = one / eps
			}
			if -one != septmp {
				vmax = one / eps
			}
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					if ttmp.Get(i-1, j-1) != t.Get(i-1, j-1) {
						vmax = one / eps
					}
					if qtmp.Get(i-1, j-1) != qsav.Get(i-1, j-1) {
						vmax = one / eps
					}
				}
			}

			//        Compute invariant subspace condition number only and compare
			//        Do not update Q
			golapack.Dlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Dlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Dtrsen('V', 'N', _select, &n, ttmp, &ldt, qtmp, &ldt, wrtmp, witmp, &m, &stmp, &septmp, work, &lwork, &iwork, &liwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label160
			}
			if -one != stmp {
				vmax = one / eps
			}
			if sep != septmp {
				vmax = one / eps
			}
			for i = 1; i <= n; i++ {
				for j = 1; j <= n; j++ {
					if ttmp.Get(i-1, j-1) != t.Get(i-1, j-1) {
						vmax = one / eps
					}
					if qtmp.Get(i-1, j-1) != qsav.Get(i-1, j-1) {
						vmax = one / eps
					}
				}
			}
			if vmax > rmax.Get(0) {
				rmax.Set(0, vmax)
				if (*ninfo)[0] == 0 {
					(*lmax)[0] = (*knt)
				}
			}
		label160:
		}
	}
}