package eig

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zget35 tests ZTRSYL, a routine for solving the Sylvester matrix
// equation
//
//    op(A)*X + ISGN*X*op(B) = scale*C,
//
// A and B are assumed to be in Schur canonical form, op() represents an
// optional transpose, and ISGN can be -1 or +1.  Scale is an output
// less than or equal to 1, chosen to avoid overflow in X.
//
// The test code verifies that the following residual is order 1:
//
//    norm(op(A)*X + ISGN*X*op(B) - scale*C) /
//        (EPS*max(norm(A),norm(B))*norm(X))
func zget35() (rmax float64, lmax, ninfo, knt int) {
	var trana, tranb mat.MatTrans
	var cone, rmul complex128
	var bignum, eps, large, one, res, res1, scale, smlnum, tnrm, two, xnrm, zero float64
	var _i, i, imla, imlad, imlb, imlc, info, isgn, itrana, itranb, j, m, n int
	var err error

	dum := vf(1)
	vm1 := vf(3)
	vm2 := vf(3)
	a := cmf(10, 10, opts)
	atmp := cmf(10, 10, opts)
	b := cmf(10, 10, opts)
	btmp := cmf(10, 10, opts)
	c := cmf(10, 10, opts)
	csav := cmf(10, 10, opts)
	ctmp := cmf(10, 10, opts)

	zero = 0.0
	one = 1.0
	two = 2.0
	large = 1.0e6
	cone = 1.0

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     Set up test case parameters
	vm1.Set(0, math.Sqrt(smlnum))
	vm1.Set(1, one)
	vm1.Set(2, large)
	vm2.Set(0, one)
	vm2.Set(1, one+two*eps)
	vm2.Set(2, two)

	mlist := []int{1, 1, 4, 4, 4, 4, 4, 4, 6}
	nlist := []int{1, 3, 4, 4, 4, 4, 4, 3, 5}
	atmplist := [][]complex128{
		{
			2.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 1.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			0.0621e0 + 0.7054e0i, 0.1062e0 + 0.0503e0i, 0.6553e0 + 0.5876e0i, 0.2560e0 + 0.8642e0i,
			0.0e0 + 0.0e0i, 0.2640e0 + 0.5782e0i, 0.9700e0 + 0.7256e0i, 0.5598e0 + 0.1943e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0380e0 + 0.2849e0i, 0.9166e0 + 0.0580e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.1402e0 + 0.6908e0i,
		},
		{
			3.0e0 + 5.0e0i, 3.0e0 + 22.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + 3.0e0i, 3.0e0 + 3.0e0i, 311.e0 + 2.0e0i,
			0.0e0 + 0.0e0i, -3.0e0 + 5.0e0i, 3.0e0 + 2.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + 3.0e0i, 11.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 3.0e0 + 2.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + 3.0e0i, 1.0e0 + -2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -33.e0 + 2.0e0i, 2.0e0 + 3.0e0i, 1.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -22.e0 + 3.0e0i, 1.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + -3.0e0i,
		},
	}
	btmplist := [][]complex128{
		{
			2.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 1.0e0i, 1.0e0 + 1.0e0i, 1.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 1.5e0 + 1.5e0i, 2.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 2.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			-1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i,
		},
		{
			-1.0e0 + 1.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i, 0.0e0 + 1.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -1.0e0 + 0.0e0i,
		},
		{
			0.6769e0 + 0.6219e0i, 0.5965e0 + 0.0505e0i, 0.7361e0 + 0.5069e0i,
			0.0e0 + 0.0e0i, 0.0726e0 + 0.7195e0i, 0.2531e0 + 0.9764e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.3481e0 + 0.5602e0i,
		},
		{
			9.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, -12.e0 + 0.0e0i, 1.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, -19.e0 + 0.0e0i, 12.e0 + 0.0e0i, 1.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 98.e0 + 0.0e0i, 11.0e0 + 0.0e0i, 3.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 13.0e0 + 0.0e0i, 11.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 13.0e0 + 0.0e0i,
		},
	}
	ctmplist := [][]complex128{
		{
			1.0e0 + 1.0e0i,
		},
		{
			2.0e0 + 1.0e0i, 2.0e0 + 1.0e0i, 9.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, 1.0e0 + 3.0e0i,
			2.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 8.0e0 + 9.0e0i, 2.0e0 + 2.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 7.0e0i, 0.0e0 + 0.0e0i, 2.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i, 0.0e0 + 0.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 1.0e0 + 0.0e0i,
		},
		{
			0.9110e0 + 0.7001e0i, 0.1821e0 + 0.5406e0i, 0.8879e0 + 0.5813e0i,
			0.0728e0 + 0.5887e0i, 0.3271e0 + 0.5647e0i, 0.3793e0 + 0.1667e0i,
			0.1729e0 + 0.6041e0i, 0.9368e0 + 0.3514e0i, 0.8149e0 + 0.3535e0i,
			0.3785e0 + 0.7924e0i, 0.6588e0 + 0.8646e0i, 0.1353e0 + 0.8362e0i,
		},
		{
			3.0e0 + -5.0e0i, 3.0e0 + 22.0e0i, 2.0e0 + 31.0e0i, 2.0e0 + 3.0e0i, 3.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, -3.0e0 + 5.0e0i, 33.e0 + 22.0e0i, 2.0e0 + 3.0e0i, -2.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -3.0e0 + 2.0e0i, 2.0e0 + 3.0e0i, 2.0e0 + -3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -33.e0 + 2.0e0i, 2.0e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, -22.e0 + 3.0e0i,
			0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + 0.0e0i, 0.0e0 + -2.0e0i,
		},
	}
	//     Begin test loop
	for _i, n = range nlist {
		m = mlist[_i]
		for i = 1; i <= m; i++ {
			for j = 1; j <= m; j++ {
				atmp.Set(i-1, j-1, atmplist[_i][(i-1)*(m)+j-1])
			}
		}
		for i = 1; i <= n; i++ {
			for j = 1; j <= n; j++ {
				btmp.Set(i-1, j-1, btmplist[_i][(i-1)*(n)+j-1])
			}
		}
		for i = 1; i <= m; i++ {
			for j = 1; j <= n; j++ {
				ctmp.Set(i-1, j-1, ctmplist[_i][(i-1)*(n)+j-1])
			}
		}
		for imla = 1; imla <= 3; imla++ {
			for imlad = 1; imlad <= 3; imlad++ {
				for imlb = 1; imlb <= 3; imlb++ {
					for imlc = 1; imlc <= 3; imlc++ {
						for itrana = 1; itrana <= 2; itrana++ {
							for itranb = 1; itranb <= 2; itranb++ {
								for isgn = -1; isgn <= 1; isgn += 2 {
									if itrana == 1 {
										trana = NoTrans
									}
									if itrana == 2 {
										trana = ConjTrans
									}
									if itranb == 1 {
										tranb = NoTrans
									}
									if itranb == 2 {
										tranb = ConjTrans
									}
									tnrm = zero
									for i = 1; i <= m; i++ {
										for j = 1; j <= m; j++ {
											a.Set(i-1, j-1, atmp.Get(i-1, j-1)*vm1.GetCmplx(imla-1))
											tnrm = math.Max(tnrm, a.GetMag(i-1, j-1))
										}
										a.Set(i-1, i-1, a.Get(i-1, i-1)*vm2.GetCmplx(imlad-1))
										tnrm = math.Max(tnrm, a.GetMag(i-1, i-1))
									}
									for i = 1; i <= n; i++ {
										for j = 1; j <= n; j++ {
											b.Set(i-1, j-1, btmp.Get(i-1, j-1)*vm1.GetCmplx(imlb-1))
											tnrm = math.Max(tnrm, b.GetMag(i-1, j-1))
										}
									}
									if tnrm == zero {
										tnrm = one
									}
									for i = 1; i <= m; i++ {
										for j = 1; j <= n; j++ {
											c.Set(i-1, j-1, ctmp.Get(i-1, j-1)*vm1.GetCmplx(imlc-1))
											csav.Set(i-1, j-1, c.Get(i-1, j-1))
										}
									}
									knt = knt + 1
									if scale, info, err = golapack.Ztrsyl(trana, tranb, isgn, m, n, a, b, c); err != nil || info != 0 {
										ninfo = ninfo + 1
									}
									xnrm = golapack.Zlange('M', m, n, c, dum)
									rmul = cone
									if xnrm > one && tnrm > one {
										if xnrm > bignum/tnrm {
											rmul = complex(math.Max(xnrm, tnrm), 0)
											rmul = cone / rmul
										}
									}
									if err = csav.Gemm(trana, NoTrans, m, n, m, rmul, a, c, complex(-scale, 0)*rmul); err != nil {
										panic(err)
									}
									if err = csav.Gemm(NoTrans, tranb, m, n, n, complex(float64(isgn), 0)*rmul, c, b, cone); err != nil {
										panic(err)
									}
									res1 = golapack.Zlange('M', m, n, csav, dum)
									res = res1 / math.Max(smlnum, math.Max(smlnum*xnrm, ((cmplx.Abs(rmul)*tnrm)*eps)*xnrm))
									if res > rmax {
										lmax = knt
										rmax = res
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return
}
