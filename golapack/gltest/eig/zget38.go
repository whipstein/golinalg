package eig

import (
	"math"
	"testing"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zget38 tests golapack.Ztrsen, a routine for estimating condition numbers of a
// cluster of eigenvalues and/or its associated right invariant subspace
//
// The test matrices are read from a file with logical unit number NIN.
func Zget38(rmax *mat.Vector, lmax, ninfo *[]int, knt *int, _t *testing.T) {
	var czero complex128
	var bignum, eps, epsin, one, s, sep, sepin, septmp, sin, smlnum, stmp, tnrm, tol, tolin, two, v, vmax, vmin, vmul, zero float64
	var _i, i, info, iscl, isrt, itmp, j, kmin, ldt, lwork, m, n, ndim int

	ldt = 20
	lwork = 2 * ldt * (10 + ldt)
	zero = 0.0
	one = 1.0
	two = 2.0
	epsin = 5.9605e-8
	czero = (0.0 + 0.0*1i)
	_select := make([]bool, 20)
	w := cvf(20)
	work := cvf(lwork)
	wtmp := cvf(20)
	result := vf(2)
	rwork := vf(20)
	val := vf(3)
	wsrt := vf(20)
	ipnt := make([]int, 20)
	iselec := make([]int, 20)
	q := cmf(20, 20, opts)
	qsav := cmf(20, 20, opts)
	qtmp := cmf(20, 20, opts)
	t := cmf(20, 20, opts)
	tmp := cmf(20, 20, opts)
	tsav := cmf(20, 20, opts)
	tsav1 := cmf(20, 20, opts)
	ttmp := cmf(20, 20, opts)

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

	nlist := []int{1, 1, 5, 5, 5, 6, 6, 4, 4, 3, 4, 4, 4, 5, 3, 4, 7, 5, 8, 3}
	ndimlist := []int{1, 1, 3, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 4, 3, 4, 2}
	isrtlist := []int{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0}
	iseleclist := [][]int{
		{1},
		{1},
		{2, 3, 4},
		{1, 3, 5},
		{2, 4},
		{3, 4, 6},
		{1, 3, 5},
		{3, 4},
		{2, 3},
		{2, 3},
		{1, 3},
		{1, 3, 4},
		{2, 3},
		{2, 3},
		{1, 2},
		{1, 3},
		{1, 4, 6, 7},
		{1, 3, 5},
		{1, 2, 3, 4},
		{2, 3},
	}
	tmplist := [][]complex128{
		{
			0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i,
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
			0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 2.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 2.0e0i, 2.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 3.0e0i, 3.0e0 + 0.0e0i, 0.0e0 + 3.0e0i, 3.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 4.0e0i, 4.0e0 + 0.0e0i, 0.0e0 + 4.0e0i, 4.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 9.5e-1i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 9.5e-1i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 9.5e-1i, 1.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 9.5e-1i,
		},
		{
			2.0e0 + 0.0e0i, 0.0e0 + -1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 1.0e0i, 2.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
		},
	}
	slist := [][]float64{
		{1.0e0, 0.0e0},
		{1.0e0, 1.0e0},
		{1.0e0, 2.9582e-31},
		{1.0e0, 1.1921e-07},
		{1.0e0, 1.0e0},
		{4.0124e-36, 3.2099e-36},
		{4.0124e-36, 3.2099e-36},
		{9.6350e-01, 3.3122e-01},
		{8.4053e-01, 7.4754e-01},
		{3.9550e-01, 2.0464e+01},
		{3.3333e-01, 1.2569e-01},
		{1.0e0, 8.2843e-01},
		{9.8985e-01, 4.1447e+00},
		{3.1088e-01, 4.6912e+00},
		{2.2361e-01, 1.0e0},
		{7.2803e-05, 1.1947e-04},
		{3.7241e-01, 5.2080e-01},
		{1.0e0, 4.5989e+00},
		{9.5269e-12, 2.9360e-11},
		{1.0e0, 2.0e0},
	}

	//     Read input data until N=0.  Assume input eigenvalues are sorted
	//     lexicographically (increasing by real part, then decreasing by
	//     imaginary part)
	for _i, n = range nlist {
		ndim = ndimlist[_i]
		isrt = isrtlist[_i]
		for i = 1; i <= ndim; i++ {
			iselec[i-1] = iseleclist[_i][i-1]
		}
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				tmp.Set(i-1, j-1, tmplist[_i][(i-1)*(n)+j-1])
			}
		}
		sin = slist[_i][0]
		sepin = slist[_i][1]

		tnrm = golapack.Zlange('M', &n, &n, tmp, &ldt, rwork)
		for iscl = 1; iscl <= 3; iscl++ {
			//        Scale input matrix
			(*knt) = (*knt) + 1
			golapack.Zlacpy('F', &n, &n, tmp, &ldt, t, &ldt)
			vmul = val.Get(iscl - 1)
			for i = 1; i <= n; i++ {
				goblas.Zdscal(n, vmul, t.CVector(0, i-1), 1)
			}
			if tnrm == zero {
				vmul = one
			}
			golapack.Zlacpy('F', &n, &n, t, &ldt, tsav, &ldt)

			//        Compute Schur form
			golapack.Zgehrd(&n, func() *int { y := 1; return &y }(), &n, t, &ldt, work.Off(0), work.Off(n+1-1), toPtr(lwork-n), &info)
			if info != 0 {
				(*lmax)[0] = (*knt)
				(*ninfo)[0] = (*ninfo)[0] + 1
				goto label200
			}

			//        Generate unitary matrix
			golapack.Zlacpy('L', &n, &n, t, &ldt, q, &ldt)
			golapack.Zunghr(&n, func() *int { y := 1; return &y }(), &n, q, &ldt, work.Off(0), work.Off(n+1-1), toPtr(lwork-n), &info)

			//        Compute Schur form
			for j = 1; j <= n-2; j++ {
				for i = j + 2; i <= n; i++ {
					t.Set(i-1, j-1, czero)
				}
			}
			golapack.Zhseqr('S', 'V', &n, func() *int { y := 1; return &y }(), &n, t, &ldt, w, q, &ldt, work, &lwork, &info)
			if info != 0 {
				(*lmax)[1] = (*knt)
				(*ninfo)[1] = (*ninfo)[1] + 1
				goto label200
			}

			//        Sort, _select eigenvalues
			for i = 1; i <= n; i++ {
				ipnt[i-1] = i
				_select[i-1] = false
			}
			if isrt == 0 {
				for i = 1; i <= n; i++ {
					wsrt.Set(i-1, w.GetRe(i-1))
				}
			} else {
				for i = 1; i <= n; i++ {
					wsrt.Set(i-1, w.GetIm(i-1))
				}
			}
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
				itmp = ipnt[i-1]
				ipnt[i-1] = ipnt[kmin-1]
				ipnt[kmin-1] = itmp
			}
			for i = 1; i <= ndim; i++ {
				_select[ipnt[iselec[i-1]-1]-1] = true
			}

			//        Compute condition numbers
			golapack.Zlacpy('F', &n, &n, q, &ldt, qsav, &ldt)
			golapack.Zlacpy('F', &n, &n, t, &ldt, tsav1, &ldt)
			golapack.Ztrsen('B', 'V', _select, &n, t, &ldt, q, &ldt, wtmp, &m, &s, &sep, work, &lwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label200
			}
			septmp = sep / vmul
			stmp = s

			//        Compute residuals
			Zhst01(&n, func() *int { y := 1; return &y }(), &n, tsav, &ldt, t, &ldt, q, &ldt, work, &lwork, rwork, result)
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
			golapack.Zlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Zlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Ztrsen('E', 'V', _select, &n, ttmp, &ldt, qtmp, &ldt, wtmp, &m, &stmp, &septmp, work, &lwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label200
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
			golapack.Zlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Zlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Ztrsen('V', 'V', _select, &n, ttmp, &ldt, qtmp, &ldt, wtmp, &m, &stmp, &septmp, work, &lwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label200
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
			golapack.Zlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Zlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Ztrsen('E', 'N', _select, &n, ttmp, &ldt, qtmp, &ldt, wtmp, &m, &stmp, &septmp, work, &lwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label200
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
			golapack.Zlacpy('F', &n, &n, tsav1, &ldt, ttmp, &ldt)
			golapack.Zlacpy('F', &n, &n, qsav, &ldt, qtmp, &ldt)
			septmp = -one
			stmp = -one
			golapack.Ztrsen('V', 'N', _select, &n, ttmp, &ldt, qtmp, &ldt, wtmp, &m, &stmp, &septmp, work, &lwork, &info)
			if info != 0 {
				(*lmax)[2] = (*knt)
				(*ninfo)[2] = (*ninfo)[2] + 1
				goto label200
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
		label200:
		}
	}
}
