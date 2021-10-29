package eig

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget37 tests ZTRSNA, a routine for estimating condition numbers of
// eigenvalues and/or right eigenvectors of a matrix.
//
// The test matrices are read from a file with logical unit number NIN.
func zget37(rmax *mat.Vector, lmax, ninfo *[]int) (knt int) {
	var bignum, eps, epsin, one, smlnum, tnrm, tol, tolin, two, v, vcmin, vmax, vmin, vmul, zero float64
	var _i, i, icmp, info, iscl, isrt, j, kmin, ldt, lwork, n int
	var err error

	zero = 0.0
	one = 1.0
	two = 2.0
	epsin = 5.9605e-8
	ldt = 20
	lwork = 2 * ldt * (10 + ldt)
	_select := make([]bool, 20)
	cdum := cvf(1)
	w := cvf(20)
	work := cvf(lwork)
	wtmp := cvf(20)
	dum := vf(1)
	rwork := vf(lwork)
	s := vf(20)
	sep := vf(20)
	sepin := vf(20)
	septmp := vf(20)
	sin := vf(20)
	stmp := vf(20)
	val := vf(3)
	wiin := vf(20)
	wrin := vf(20)
	wsrt := vf(20)
	lcmp := make([]int, 3)
	le := cmf(20, 20, opts)
	re := cmf(20, 20, opts)
	t := cmf(20, 20, opts)
	tmp := cmf(20, 20, opts)

	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     EPSIN = 2**(-24) = precision to which input data computed
	eps = math.Max(eps, epsin)
	rmax.Set(0, zero)
	rmax.Set(1, zero)
	rmax.Set(2, zero)
	(*lmax)[0] = 0
	(*lmax)[1] = 0
	(*lmax)[2] = 0
	(*ninfo)[0] = 0
	(*ninfo)[1] = 0
	(*ninfo)[2] = 0
	val.Set(0, math.Sqrt(smlnum))
	val.Set(1, one)
	val.Set(2, math.Sqrt(bignum))

	nlist := []int{1, 1, 2, 2, 2, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 3}
	isrtlist := []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0}
	tmplist := [][]complex128{
		{
			0.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 1.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			3.0e0 + 0.0e0i, 2.0e0 + 0.0e0i,
			2.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
		},
		{
			3.0e0 + 0.0e0i, 0.0e0 + 2.0e0i,
			0.0e0 + 2.0e0i, 3.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 4.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 5.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i,
		},
		{
			0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i,
		},
		{
			9.4480e-01 + 1.0e0i, 6.7670e-01 + 1.0e0i, 6.9080e-01 + 1.0e0i, 5.9650e-01 + 1.0e0i,
			5.8760e-01 + 1.0e0i, 8.6420e-01 + 1.0e0i, 6.7690e-01 + 1.0e0i, 7.2600e-02 + 1.0e0i,
			7.2560e-01 + 1.0e0i, 1.9430e-01 + 1.0e0i, 9.6870e-01 + 1.0e0i, 2.8310e-01 + 1.0e0i,
			2.8490e-01 + 1.0e0i, 5.8000e-02 + 1.0e0i, 4.8450e-01 + 1.0e0i, 7.3610e-01 + 1.0e0i,
		},
		{
			2.1130e-01 + 9.9330e-01i, 8.0960e-01 + 4.2370e-01i, 4.8320e-01 + 1.1670e-01i, 6.5380e-01 + 4.9430e-01i,
			8.2400e-02 + 8.3600e-01i, 8.4740e-01 + 2.6130e-01i, 6.1350e-01 + 6.2500e-01i, 4.8990e-01 + 3.6500e-02i,
			7.5990e-01 + 7.4690e-01i, 4.5240e-01 + 2.4030e-01i, 2.7490e-01 + 5.5100e-01i, 7.7410e-01 + 2.2600e-01i,
			8.7000e-03 + 3.7800e-02i, 8.0750e-01 + 3.4050e-01i, 8.8070e-01 + 3.5500e-01i, 9.6260e-01 + 8.1590e-01i,
		},
		{
			1.0e0 + 2.0e0i, 3.0e0 + 4.0e0i, 2.1e1 + 2.2e1i,
			4.3e1 + 4.4e1i, 1.3e1 + 1.4e1i, 1.5e1 + 1.6e1i,
			5.0e0 + 6.0e0i, 7.0e0 + 8.0e0i, 2.5e1 + 2.6e1i,
		},
		{
			5.0e0 + 9.0e0i, 5.0e0 + 5.0e0i, -6.0e0 + -6.0e0i, -7.0e0 + -7.0e0i,
			3.0e0 + 3.0e0i, 6.0e0 + 1.0e1i, -5.0e0 + -5.0e0i, -6.0e0 + -6.0e0i,
			2.0e0 + 2.0e0i, 3.0e0 + 3.0e0i, -1.0e0 + 3.0e0i, -5.0e0 + -5.0e0i,
			1.0e0 + 1.0e0i, 2.0e0 + 2.0e0i, -3.0e0 + -3.0e0i, 0.0e0 + 4.0e0i,
		},
		{
			3.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 2.0e0i,
			1.0e0 + 0.0e0i, 3.0e0 + 0.0e0i, 0.0e0 + -2.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 2.0e0i, 1.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
			0.0e0 + -2.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			7.0e0 + 0.0e0i, 3.0e0 + 0.0e0i, 1.0e0 + 2.0e0i, -1.0e0 + 2.0e0i,
			3.0e0 + 0.0e0i, 7.0e0 + 0.0e0i, 1.0e0 + -2.0e0i, -1.0e0 + -2.0e0i,
			1.0e0 + -2.0e0i, 1.0e0 + 2.0e0i, 7.0e0 + 0.0e0i, -3.0e0 + 0.0e0i,
			-1.0e0 + -2.0e0i, -2.0e0 + 2.0e0i, -3.0e0 + 0.0e0i, 7.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 2.0e0i, 3.0e0 + 4.0e0i, 2.1e1 + 2.2e1i, 2.3e1 + 2.4e1i, 4.1e1 + 4.2e1i,
			4.3e1 + 4.4e1i, 1.3e1 + 1.4e1i, 1.5e1 + 1.6e1i, 3.3e1 + 3.4e1i, 3.5e1 + 3.6e1i,
			5.0e0 + 6.0e0i, 7.0e0 + 8.0e0i, 2.5e1 + 2.6e1i, 2.7e1 + 2.8e1i, 4.5e1 + 4.6e1i,
			4.7e1 + 4.8e1i, 1.7e1 + 1.8e1i, 1.9e1 + 2.0e1i, 3.7e1 + 3.8e1i, 3.9e1 + 4.0e1i,
			9.0e0 + 1.0e1i, 1.1e1 + 1.2e1i, 2.9e1 + 3.0e1i, 3.1e1 + 3.2e1i, 4.9e1 + 5.0e1i,
		},
		{
			1.0e0 + 1.0e0i, -1.0e0 + -1.0e0i, 2.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 2.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i, 3.0e0 + 1.0e0i,
		},
		{
			-4.0e0 + -2.0e0i, -5.0e0 + -6.0e0i, -2.0e0 + -6.0e0i, 0.0e0 + -2.0e0i,
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			2.0e0 + 4.0e0i, 1.0e0 + 1.0e0i, 6.0e0 + 2.0e0i, 3.0e0 + 3.0e0i, 5.0e0 + 5.0e0i, 2.0e0 + 6.0e0i, 1.0e0 + 1.0e0i,
			1.0e0 + 2.0e0i, 1.0e0 + 3.0e0i, 3.0e0 + 1.0e0i, 5.0e0 + -4.0e0i, 1.0e0 + 1.0e0i, 7.0e0 + 2.0e0i, 2.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 3.0e0 + -2.0e0i, 1.0e0 + 1.0e0i, 6.0e0 + 3.0e0i, 2.0e0 + 1.0e0i, 1.0e0 + 4.0e0i, 2.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 3.0e0i, 3.0e0 + 1.0e0i, 1.0e0 + 2.0e0i, 2.0e0 + 2.0e0i, 3.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + -1.0e0i, 2.0e0 + 2.0e0i, 3.0e0 + 1.0e0i, 1.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + -1.0e0i, 2.0e0 + 1.0e0i, 2.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + -2.0e0i, 1.0e0 + 1.0e0i,
		},
		{
			0.0e0 + 5.0e0i, 1.0e0 + 2.0e0i, 2.0e0 + 3.0e0i, -3.0e0 + 6.0e0i, 6.0e0 + 0.0e0i,
			-1.0e0 + 2.0e0i, 0.0e0 + 6.0e0i, 4.0e0 + 5.0e0i, -3.0e0 + -2.0e0i, 5.0e0 + 0.0e0i,
			-2.0e0 + 3.0e0i, -4.0e0 + 5.0e0i, 0.0e0 + 7.0e0i, 3.0e0 + 0.0e0i, 2.0e0 + 0.0e0i,
			3.0e0 + 6.0e0i, 3.0e0 + -2.0e0i, -3.0e0 + 0.0e0i, 0.0e0 + -5.0e0i, 2.0e0 + 1.0e0i,
			-6.0e0 + 0.0e0i, -5.0e0 + 0.0e0i, -2.0e0 + 0.0e0i, -2.0e0 + 1.0e0i, 0.0e0 + 2.0e0i,
		},
		{
			2.0e0 + 0.0e0i, 0.0e0 + -1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 1.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
		},
	}
	wlist := [][]float64{
		{
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
		},
		{
			0.0e0, 1.0e0, 1.0e0, 1.0e0,
		},
		{
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
		},
		{
			1.0e0, 0.0e0, 1.0e0, 4.0e0,
			5.0e0, 0.0e0, 1.0e0, 4.0e0,
		},
		{
			3.0e0, 2.0e0, 1.0e0, 4.0e0,
			3.0e0, -2.0e0, 1.0e0, 4.0e0,
		},
		{
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
			0.0e0, 0.0e0, 1.0e0, 0.0e0,
		},
		{
			1.0e0, 0.0e0, 1.0e0, 0.0e0,
			1.0e0, 0.0e0, 1.0e0, 0.0e0,
			1.0e0, 0.0e0, 1.0e0, 0.0e0,
			1.0e0, 0.0e0, 1.0e0, 0.0e0,
			1.0e0, 0.0e0, 1.0e0, 0.0e0,
		},
		{
			1.0e0, 0.0e0, 1.0e0, 1.0e0,
			2.0e0, 0.0e0, 1.0e0, 1.0e0,
			3.0e0, 0.0e0, 1.0e0, 1.0e0,
			4.0e0, 0.0e0, 1.0e0, 1.0e0,
			5.0e0, 0.0e0, 1.0e0, 1.0e0,
		},
		{
			0.0e0, 1.0e0, 1.1921e-07, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 1.1921e-07, 0.0e0,
		},
		{
			0.0e0, 1.0e0, 1.1921e-07, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 2.4074e-35, 0.0e0,
			0.0e0, 1.0e0, 1.1921e-07, 0.0e0,
		},
		{
			2.6014e-01, -1.7813e-01, 8.5279e-01, 3.2881e-01,
			2.8961e-01, 2.0772e-01, 8.4871e-01, 3.2358e-01,
			7.3990e-01, -4.6522e-04, 9.7398e-01, 3.4994e-01,
			2.2242e+00, 3.9709e+00, 9.8325e-01, 4.1429e+00,
		},
		{
			-6.2157e-01, 6.0607e-01, 8.7533e-01, 8.1980e-01,
			2.8890e-01, -2.6354e-01, 8.2538e-01, 8.1086e-01,
			3.8017e-01, 5.4217e-01, 7.4771e-01, 7.0323e-01,
			2.2487e+00, 1.7368e+00, 9.2372e-01, 2.2178e+00,
		},
		{
			-7.4775e+00, 6.8803e+00, 3.9550e-01, 1.6583e+01,
			6.7009e+00, -7.8760e+00, 3.9828e-01, 1.6312e+01,
			3.9777e+01, 4.2996e+01, 7.9686e-01, 3.7399e+01,
		},
		{
			1.0e0, 5.0e0, 2.1822e-01, 7.4651e-01,
			2.0e0, 6.0e0, 2.1822e-01, 3.0893e-01,
			3.0e0, 7.0e0, 2.1822e-01, 1.8315e-01,
			4.0e0, 8.0e0, 2.1822e-01, 6.6350e-01,
		},
		{
			-8.2843e-01, 1.6979e-07, 1.0e0, 8.2843e-01,
			4.1744e-07, 7.1526e-08, 1.0e0, 8.2843e-01,
			4.0e0, 1.6690e-07, 1.0e0, 8.2843e-01,
			4.8284e+00, 6.8633e-08, 1.0e0, 8.2843e-01,
		},
		{
			-8.0767e-03, -2.5211e-01, 9.9864e-01, 7.7961e+00,
			7.7723e+00, 2.4349e-01, 7.0272e-01, 3.3337e-01,
			8.0e0, -3.4273e-07, 7.0711e-01, 3.3337e-01,
			1.2236e+01, 8.6188e-03, 9.9021e-01, 3.9429e+00,
		},
		{
			-9.4600e+00, 7.2802e+00, 3.1053e-01, 1.1937e+01,
			-7.7912e-06, -1.2743e-05, 2.9408e-01, 1.6030e-05,
			-7.3042e-06, 3.2789e-06, 7.2259e-01, 6.7794e-06,
			7.0733e+00, -9.5584e+00, 3.0911e-01, 1.1891e+01,
			1.2739e+02, 1.3228e+02, 9.2770e-01, 1.2111e+02,
		},
		{
			1.0e0, 1.0e0, 3.0151e-01, 0.0e0,
			1.0e0, 1.0e0, 3.1623e-01, 0.0e0,
			2.0e0, 1.0e0, 2.2361e-01, 1.0e0,
		},
		{
			-9.9883e-01, -1.0006e+00, 1.3180e-04, 2.4106e-04,
			-1.0012e+00, -9.9945e-01, 1.3140e-04, 2.4041e-04,
			-9.9947e-01, -6.8325e-04, 1.3989e-04, 8.7487e-05,
			-1.0005e+00, 6.8556e-04, 1.4010e-04, 8.7750e-05,
		},
		{
			-2.7081e+00, -2.8029e+00, 6.9734e-01, 3.9279e+00,
			-1.1478e+00, 8.0176e-01, 6.5772e-01, 9.4243e-01,
			-8.0109e-01, 4.9694e+00, 4.6751e-01, 1.3779e+00,
			9.9492e-01, 3.1688e+00, 3.5095e-01, 5.9845e-01,
			2.0809e+00, 1.9341e+00, 4.9042e-01, 3.9035e-01,
			5.3138e+00, 1.2242e+00, 3.0213e-01, 7.1268e-01,
			8.2674e+00, 3.7047e+00, 2.8270e-01, 3.2849e+00,
		},
		{
			-4.1735e-08, -1.0734e+01, 1.0e0, 7.7345e+00,
			-2.6397e-07, -2.9991e+00, 1.0e0, 4.5989e+00,
			1.4565e-07, 1.5998e+00, 1.0e0, 4.5989e+00,
			-4.4369e-07, 9.3159e+00, 1.0e0, 7.7161e+00,
			4.0937e-09, 1.7817e+01, 1.0e0, 8.5013e+00,
		},
		{
			1.0e0, 0.0e0, 1.0e0, 2.0e0,
			3.0e0, 0.0e0, 1.0e0, 0.0e0,
			3.0e0, 0.0e0, 1.0e0, 0.0e0,
		},
	}

	//     Read input data until N=0.  Assume input eigenvalues are sorted
	//     lexicographically (increasing by real part if ISRT = 0,
	//     increasing by imaginary part if ISRT = 1)
	for _i, n = range nlist {
		isrt = isrtlist[_i]
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				tmp.Set(i-1, j-1, tmplist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			wrin.Set(i-1, wlist[_i][(i-1)*4+0])
			wiin.Set(i-1, wlist[_i][(i-1)*4+1])
			sin.Set(i-1, wlist[_i][(i-1)*4+2])
			sepin.Set(i-1, wlist[_i][(i-1)*4+3])
		}
		tnrm = golapack.Zlange('M', n, n, tmp, rwork)
		for iscl = 1; iscl <= 3; iscl++ {
			//        Scale input matrix
			knt = knt + 1
			golapack.Zlacpy(Full, n, n, tmp, t)
			vmul = val.Get(iscl - 1)
			for i = 1; i <= n; i++ {
				goblas.Zdscal(n, vmul, t.CVector(0, i-1, 1))
			}
			if tnrm == zero {
				vmul = one
			}

			//        Compute eigenvalues and eigenvectors
			if err = golapack.Zgehrd(n, 1, n, t, work.Off(0), work.Off(n), lwork-n); err != nil {
				(*lmax)[0] = knt
				(*ninfo)[0] = (*ninfo)[0] + 1
				goto label260
			}
			for j = 1; j <= n-2; j++ {
				for i = j + 2; i <= n; i++ {
					t.SetRe(i-1, j-1, zero)
				}
			}

			//        Compute Schur form
			if info, err = golapack.Zhseqr('S', 'N', n, 1, n, t, w, cdum.CMatrix(1, opts), work, lwork); err != nil || info != 0 {
				(*lmax)[1] = knt
				(*ninfo)[1] = (*ninfo)[1] + 1
				goto label260
			}

			//        Compute eigenvectors
			for i = 1; i <= n; i++ {
				_select[i-1] = true
			}
			if _, err = golapack.Ztrevc(Both, 'A', _select, n, t, le, re, n, work, rwork); err != nil {
				panic(err)
			}

			//        Compute condition numbers
			if _, err = golapack.Ztrsna('B', 'A', _select, n, t, le, re, s, sep, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}

			//        Sort eigenvalues and condition numbers lexicographically
			//        to compare with inputs
			goblas.Zcopy(n, w.Off(0, 1), wtmp.Off(0, 1))
			if isrt == 0 {
				//           Sort by increasing real part
				for i = 1; i <= n; i++ {
					wsrt.Set(i-1, w.GetRe(i-1))
				}
			} else {
				//           Sort by increasing imaginary part
				for i = 1; i <= n; i++ {
					wsrt.Set(i-1, w.GetIm(i-1))
				}
			}
			goblas.Dcopy(n, s.Off(0, 1), stmp.Off(0, 1))
			goblas.Dcopy(n, sep.Off(0, 1), septmp.Off(0, 1))
			goblas.Dscal(n, one/vmul, septmp.Off(0, 1))
			for i = 1; i <= n-1; i++ {
				kmin = i
				vmin = wsrt.Get(i - 1)
				for j = i + 1; j <= n; j++ {
					if wsrt.Get(j-1) < vmin {
						kmin = j
						vmin = wsrt.Get(j - 1)
					}
				}
				wsrt.Set(kmin-1, wsrt.Get(i-1))
				wsrt.Set(i-1, vmin)
				vcmin = wtmp.GetRe(i - 1)
				wtmp.Set(i-1, w.Get(kmin-1))
				wtmp.SetRe(kmin-1, vcmin)
				vmin = stmp.Get(kmin - 1)
				stmp.Set(kmin-1, stmp.Get(i-1))
				stmp.Set(i-1, vmin)
				vmin = septmp.Get(kmin - 1)
				septmp.Set(kmin-1, septmp.Get(i-1))
				septmp.Set(i-1, vmin)
			}

			//        Compare condition numbers for eigenvalues
			//        taking their condition numbers into account
			v = math.Max(two*float64(n)*eps*tnrm, smlnum)
			if tnrm == zero {
				v = one
			}
			for i = 1; i <= n; i++ {
				if v > septmp.Get(i-1) {
					tol = one
				} else {
					tol = v / septmp.Get(i-1)
				}
				if v > sepin.Get(i-1) {
					tolin = one
				} else {
					tolin = v / sepin.Get(i-1)
				}
				tol = math.Max(tol, smlnum/eps)
				tolin = math.Max(tolin, smlnum/eps)
				if eps*(sin.Get(i-1)-tolin) > stmp.Get(i-1)+tol {
					vmax = one / eps
				} else if sin.Get(i-1)-tolin > stmp.Get(i-1)+tol {
					vmax = (sin.Get(i-1) - tolin) / (stmp.Get(i-1) + tol)
				} else if sin.Get(i-1)+tolin < eps*(stmp.Get(i-1)-tol) {
					vmax = one / eps
				} else if sin.Get(i-1)+tolin < stmp.Get(i-1)-tol {
					vmax = (stmp.Get(i-1) - tol) / (sin.Get(i-1) + tolin)
				} else {
					vmax = one
				}
				if vmax > rmax.Get(1) {
					rmax.Set(1, vmax)
					if (*ninfo)[1] == 0 {
						(*lmax)[1] = knt
					}
				}
			}

			//        Compare condition numbers for eigenvectors
			//        taking their condition numbers into account
			for i = 1; i <= n; i++ {
				if v > septmp.Get(i-1)*stmp.Get(i-1) {
					tol = septmp.Get(i - 1)
				} else {
					tol = v / stmp.Get(i-1)
				}
				if v > sepin.Get(i-1)*sin.Get(i-1) {
					tolin = sepin.Get(i - 1)
				} else {
					tolin = v / sin.Get(i-1)
				}
				tol = math.Max(tol, smlnum/eps)
				tolin = math.Max(tolin, smlnum/eps)
				if eps*(sepin.Get(i-1)-tolin) > septmp.Get(i-1)+tol {
					vmax = one / eps
				} else if sepin.Get(i-1)-tolin > septmp.Get(i-1)+tol {
					vmax = (sepin.Get(i-1) - tolin) / (septmp.Get(i-1) + tol)
				} else if sepin.Get(i-1)+tolin < eps*(septmp.Get(i-1)-tol) {
					vmax = one / eps
				} else if sepin.Get(i-1)+tolin < septmp.Get(i-1)-tol {
					vmax = (septmp.Get(i-1) - tol) / (sepin.Get(i-1) + tolin)
				} else {
					vmax = one
				}
				if vmax > rmax.Get(1) {
					rmax.Set(1, vmax)
					if (*ninfo)[1] == 0 {
						(*lmax)[1] = knt
					}
				}
			}

			//        Compare condition numbers for eigenvalues
			//        without taking their condition numbers into account
			for i = 1; i <= n; i++ {
				if sin.Get(i-1) <= float64(2*n)*eps && stmp.Get(i-1) <= float64(2*n)*eps {
					vmax = one
				} else if eps*sin.Get(i-1) > stmp.Get(i-1) {
					vmax = one / eps
				} else if sin.Get(i-1) > stmp.Get(i-1) {
					vmax = sin.Get(i-1) / stmp.Get(i-1)
				} else if sin.Get(i-1) < eps*stmp.Get(i-1) {
					vmax = one / eps
				} else if sin.Get(i-1) < stmp.Get(i-1) {
					vmax = stmp.Get(i-1) / sin.Get(i-1)
				} else {
					vmax = one
				}
				if vmax > rmax.Get(2) {
					rmax.Set(2, vmax)
					if (*ninfo)[2] == 0 {
						(*lmax)[2] = knt
					}
				}
			}

			//        Compare condition numbers for eigenvectors
			//        without taking their condition numbers into account
			for i = 1; i <= n; i++ {
				if sepin.Get(i-1) <= v && septmp.Get(i-1) <= v {
					vmax = one
				} else if eps*sepin.Get(i-1) > septmp.Get(i-1) {
					vmax = one / eps
				} else if sepin.Get(i-1) > septmp.Get(i-1) {
					vmax = sepin.Get(i-1) / septmp.Get(i-1)
				} else if sepin.Get(i-1) < eps*septmp.Get(i-1) {
					vmax = one / eps
				} else if sepin.Get(i-1) < septmp.Get(i-1) {
					vmax = septmp.Get(i-1) / sepin.Get(i-1)
				} else {
					vmax = one
				}
				if vmax > rmax.Get(2) {
					rmax.Set(2, vmax)
					if (*ninfo)[2] == 0 {
						(*lmax)[2] = knt
					}
				}
			}

			//        Compute eigenvalue condition numbers only and compare
			vmax = zero
			dum.Set(0, -one)
			goblas.Dcopy(n, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(n, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('E', 'A', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= n; i++ {
				if stmp.Get(i-1) != s.Get(i-1) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
			}

			//        Compute eigenvector condition numbers only and compare
			goblas.Dcopy(n, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(n, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('V', 'A', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= n; i++ {
				if stmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != sep.Get(i-1) {
					vmax = one / eps
				}
			}

			//        Compute all condition numbers using SELECT and compare
			for i = 1; i <= n; i++ {
				_select[i-1] = true
			}
			goblas.Dcopy(n, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(n, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('B', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= n; i++ {
				if septmp.Get(i-1) != sep.Get(i-1) {
					vmax = one / eps
				}
				if stmp.Get(i-1) != s.Get(i-1) {
					vmax = one / eps
				}
			}

			//        Compute eigenvalue condition numbers using SELECT and compare
			goblas.Dcopy(n, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(n, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('E', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= n; i++ {
				if stmp.Get(i-1) != s.Get(i-1) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
			}

			//        Compute eigenvector condition numbers using SELECT and compare
			goblas.Dcopy(n, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(n, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('V', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= n; i++ {
				if stmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != sep.Get(i-1) {
					vmax = one / eps
				}
			}
			if vmax > rmax.Get(0) {
				rmax.Set(0, vmax)
				if (*ninfo)[0] == 0 {
					(*lmax)[0] = knt
				}
			}

			//        Select second and next to last eigenvalues
			for i = 1; i <= n; i++ {
				_select[i-1] = false
			}
			icmp = 0
			if n > 1 {
				icmp = 1
				lcmp[0] = 2
				_select[1] = true
				goblas.Zcopy(n, re.CVector(0, 1, 1), re.CVector(0, 0, 1))
				goblas.Zcopy(n, le.CVector(0, 1, 1), le.CVector(0, 0, 1))
			}
			if n > 3 {
				icmp = 2
				lcmp[1] = n - 1
				_select[n-1-1] = true
				goblas.Zcopy(n, re.CVector(0, n-1-1, 1), re.CVector(0, 1, 1))
				goblas.Zcopy(n, le.CVector(0, n-1-1, 1), le.CVector(0, 1, 1))
			}

			//        Compute all selected condition numbers
			goblas.Dcopy(icmp, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(icmp, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('B', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= icmp; i++ {
				j = lcmp[i-1]
				if septmp.Get(i-1) != sep.Get(j-1) {
					vmax = one / eps
				}
				if stmp.Get(i-1) != s.Get(j-1) {
					vmax = one / eps
				}
			}

			//        Compute selected eigenvalue condition numbers
			goblas.Dcopy(icmp, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(icmp, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('E', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= icmp; i++ {
				j = lcmp[i-1]
				if stmp.Get(i-1) != s.Get(j-1) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
			}

			//        Compute selected eigenvector condition numbers
			goblas.Dcopy(icmp, dum.Off(0, 0), stmp.Off(0, 1))
			goblas.Dcopy(icmp, dum.Off(0, 0), septmp.Off(0, 1))
			if _, err = golapack.Ztrsna('V', 'S', _select, n, t, le, re, stmp, septmp, n, work.CMatrix(n, opts), rwork); err != nil {
				(*lmax)[2] = knt
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label260
			}
			for i = 1; i <= icmp; i++ {
				j = lcmp[i-1]
				if stmp.Get(i-1) != dum.Get(0) {
					vmax = one / eps
				}
				if septmp.Get(i-1) != sep.Get(j-1) {
					vmax = one / eps
				}
			}
			if vmax > rmax.Get(0) {
				rmax.Set(0, vmax)
				if (*ninfo)[0] == 0 {
					(*lmax)[0] = knt
				}
			}
		label260:
		}
	}

	return
}
