package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// dget39 tests DLAQTR, a routine for solving the real or
// special complex quasi upper triangular system
//
//      op(T)*p = scale*c,
// or
//      op(T + iB)*(p+iq) = scale*(c+id),
//
// in real arithmetic. T is upper quasi-triangular.
// If it is complex, then the first diagonal block of T must be
// 1 by 1, B has the special structure
//
//                B = [ b(1) b(2) ... b(n) ]
//                    [       w            ]
//                    [           w        ]
//                    [              .     ]
//                    [                 w  ]
//
// op(A) = A or A', where A' denotes the conjugate transpose of
// the matrix A.
//
// On input, X = [ c ].  On output, X = [ p ].
//               [ d ]                  [ q ]
//
// Scale is an output less than or equal to 1, chosen to avoid
// overflow in X.
// This subroutine is specially designed for the condition number
// estimation in the eigenproblem routine DTRSNA.
//
// The test code verifies that the following residual is order 1:
//
//      ||(T+i*B)*(x1+i*x2) - scale*(d1+i*d2)||
//    -----------------------------------------
//        max(ulp*(||T||+||B||)*(||x1||+||x2||),
//            (||T||+||B||)*smlnum/ulp,
//            smlnum)
//
// (The (||T||+||B||)*smlnum/ulp term accounts for possible
//  (gradual or nongradual) underflow in x1 and x2.)
func dget39() (rmax float64, lmax, ninfo, knt int) {
	var bignum, domin, dumm, eps, norm, normtb, one, resid, scale, smlnum, w, xnorm, zero float64
	var i, info, ivm1, ivm2, ivm3, ivm4, ivm5, j, k, ldt, ldt2, n, ndim int
	var err error

	ldt = 10
	ldt2 = 2 * ldt
	zero = 0.0
	one = 1.0
	b := vf(10)
	d := vf(ldt2)
	dum := vf(1)
	vm1 := vf(5)
	vm2 := vf(5)
	vm3 := vf(5)
	vm4 := vf(5)
	vm5 := vf(3)
	work := vf(10)
	x := vf(ldt2)
	y := vf(ldt2)
	idim := make([]int, 6)
	t := mf(10, 10, opts)
	ival := make([]int, 5*5*6)

	idim[0], idim[1], idim[2], idim[3], idim[4], idim[5] = 4, 5, 5, 5, 5, 5
	ival[0+(0+(0)*5)*5], ival[1+(0+(0)*5)*5], ival[2+(0+(0)*5)*5], ival[3+(0+(0)*5)*5], ival[4+(0+(0)*5)*5], ival[0+(1+(0)*5)*5], ival[1+(1+(0)*5)*5], ival[2+(1+(0)*5)*5], ival[3+(1+(0)*5)*5], ival[4+(1+(0)*5)*5], ival[0+(2+(0)*5)*5], ival[1+(2+(0)*5)*5], ival[2+(2+(0)*5)*5], ival[3+(2+(0)*5)*5], ival[4+(2+(0)*5)*5], ival[0+(3+(0)*5)*5], ival[1+(3+(0)*5)*5], ival[2+(3+(0)*5)*5], ival[3+(3+(0)*5)*5], ival[4+(3+(0)*5)*5], ival[0+(4+(0)*5)*5], ival[1+(4+(0)*5)*5], ival[2+(4+(0)*5)*5], ival[3+(4+(0)*5)*5], ival[4+(4+(0)*5)*5], ival[0+(0+(1)*5)*5], ival[1+(0+(1)*5)*5], ival[2+(0+(1)*5)*5], ival[3+(0+(1)*5)*5], ival[4+(0+(1)*5)*5], ival[0+(1+(1)*5)*5], ival[1+(1+(1)*5)*5], ival[2+(1+(1)*5)*5], ival[3+(1+(1)*5)*5], ival[4+(1+(1)*5)*5], ival[0+(2+(1)*5)*5], ival[1+(2+(1)*5)*5], ival[2+(2+(1)*5)*5], ival[3+(2+(1)*5)*5], ival[4+(2+(1)*5)*5], ival[0+(3+(1)*5)*5], ival[1+(3+(1)*5)*5], ival[2+(3+(1)*5)*5], ival[3+(3+(1)*5)*5], ival[4+(3+(1)*5)*5], ival[0+(4+(1)*5)*5], ival[1+(4+(1)*5)*5], ival[2+(4+(1)*5)*5], ival[3+(4+(1)*5)*5], ival[4+(4+(1)*5)*5], ival[0+(0+(2)*5)*5], ival[1+(0+(2)*5)*5], ival[2+(0+(2)*5)*5], ival[3+(0+(2)*5)*5], ival[4+(0+(2)*5)*5], ival[0+(1+(2)*5)*5], ival[1+(1+(2)*5)*5], ival[2+(1+(2)*5)*5], ival[3+(1+(2)*5)*5], ival[4+(1+(2)*5)*5], ival[0+(2+(2)*5)*5], ival[1+(2+(2)*5)*5], ival[2+(2+(2)*5)*5], ival[3+(2+(2)*5)*5], ival[4+(2+(2)*5)*5], ival[0+(3+(2)*5)*5], ival[1+(3+(2)*5)*5], ival[2+(3+(2)*5)*5], ival[3+(3+(2)*5)*5], ival[4+(3+(2)*5)*5], ival[0+(4+(2)*5)*5], ival[1+(4+(2)*5)*5], ival[2+(4+(2)*5)*5], ival[3+(4+(2)*5)*5], ival[4+(4+(2)*5)*5], ival[0+(0+(3)*5)*5], ival[1+(0+(3)*5)*5], ival[2+(0+(3)*5)*5], ival[3+(0+(3)*5)*5], ival[4+(0+(3)*5)*5], ival[0+(1+(3)*5)*5], ival[1+(1+(3)*5)*5], ival[2+(1+(3)*5)*5], ival[3+(1+(3)*5)*5], ival[4+(1+(3)*5)*5], ival[0+(2+(3)*5)*5], ival[1+(2+(3)*5)*5], ival[2+(2+(3)*5)*5], ival[3+(2+(3)*5)*5], ival[4+(2+(3)*5)*5], ival[0+(3+(3)*5)*5], ival[1+(3+(3)*5)*5], ival[2+(3+(3)*5)*5], ival[3+(3+(3)*5)*5], ival[4+(3+(3)*5)*5], ival[0+(4+(3)*5)*5], ival[1+(4+(3)*5)*5], ival[2+(4+(3)*5)*5], ival[3+(4+(3)*5)*5], ival[4+(4+(3)*5)*5], ival[0+(0+(4)*5)*5], ival[1+(0+(4)*5)*5], ival[2+(0+(4)*5)*5], ival[3+(0+(4)*5)*5], ival[4+(0+(4)*5)*5], ival[0+(1+(4)*5)*5], ival[1+(1+(4)*5)*5], ival[2+(1+(4)*5)*5], ival[3+(1+(4)*5)*5], ival[4+(1+(4)*5)*5], ival[0+(2+(4)*5)*5], ival[1+(2+(4)*5)*5], ival[2+(2+(4)*5)*5], ival[3+(2+(4)*5)*5], ival[4+(2+(4)*5)*5], ival[0+(3+(4)*5)*5], ival[1+(3+(4)*5)*5], ival[2+(3+(4)*5)*5], ival[3+(3+(4)*5)*5], ival[4+(3+(4)*5)*5], ival[0+(4+(4)*5)*5], ival[1+(4+(4)*5)*5], ival[2+(4+(4)*5)*5], ival[3+(4+(4)*5)*5], ival[4+(4+(4)*5)*5], ival[0+(0+(5)*5)*5], ival[1+(0+(5)*5)*5], ival[2+(0+(5)*5)*5], ival[3+(0+(5)*5)*5], ival[4+(0+(5)*5)*5], ival[0+(1+(5)*5)*5], ival[1+(1+(5)*5)*5], ival[2+(1+(5)*5)*5], ival[3+(1+(5)*5)*5], ival[4+(1+(5)*5)*5], ival[0+(2+(5)*5)*5], ival[1+(2+(5)*5)*5], ival[2+(2+(5)*5)*5], ival[3+(2+(5)*5)*5], ival[4+(2+(5)*5)*5], ival[0+(3+(5)*5)*5], ival[1+(3+(5)*5)*5], ival[2+(3+(5)*5)*5], ival[3+(3+(5)*5)*5], ival[4+(3+(5)*5)*5], ival[0+(4+(5)*5)*5], ival[1+(4+(5)*5)*5], ival[2+(4+(5)*5)*5], ival[3+(4+(5)*5)*5], ival[4+(4+(5)*5)*5] = 3, 0, 0, 0, 0, 1, 1, -1, 0, 0, 3, 2, 1, 0, 0, 4, 3, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 3, 3, 4, 0, 0, 4, 2, 2, 3, 0, 1, 1, 1, 1, 5, 1, 0, 0, 0, 0, 2, 4, -2, 0, 0, 3, 3, 4, 0, 0, 4, 2, 2, 3, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 1, -1, 0, 0, 9, 8, 1, 0, 0, 4, 9, 1, 2, -1, 2, 2, 2, 2, 2, 9, 0, 0, 0, 0, 6, 4, 0, 0, 0, 3, 2, 1, 1, 0, 5, 1, -1, 1, 0, 2, 2, 2, 2, 2, 4, 0, 0, 0, 0, 2, 2, 0, 0, 0, 1, 4, 4, 0, 0, 2, 4, 2, 2, -1, 2, 2, 2, 2, 2

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)

	//     Set up test case parameters
	vm1.Set(0, one)
	vm1.Set(1, math.Sqrt(smlnum))
	vm1.Set(2, math.Sqrt(vm1.Get(1)))
	vm1.Set(3, math.Sqrt(bignum))
	vm1.Set(4, math.Sqrt(vm1.Get(3)))

	vm2.Set(0, one)
	vm2.Set(1, math.Sqrt(smlnum))
	vm2.Set(2, math.Sqrt(vm2.Get(1)))
	vm2.Set(3, math.Sqrt(bignum))
	vm2.Set(4, math.Sqrt(vm2.Get(3)))

	vm3.Set(0, one)
	vm3.Set(1, math.Sqrt(smlnum))
	vm3.Set(2, math.Sqrt(vm3.Get(1)))
	vm3.Set(3, math.Sqrt(bignum))
	vm3.Set(4, math.Sqrt(vm3.Get(3)))

	vm4.Set(0, one)
	vm4.Set(1, math.Sqrt(smlnum))
	vm4.Set(2, math.Sqrt(vm4.Get(1)))
	vm4.Set(3, math.Sqrt(bignum))
	vm4.Set(4, math.Sqrt(vm4.Get(3)))

	vm5.Set(0, one)
	vm5.Set(1, eps)
	vm5.Set(2, math.Sqrt(smlnum))

	//     Initialization
	knt = 0
	rmax = zero
	ninfo = 0
	smlnum = smlnum / eps

	//     Begin test loop
	for ivm5 = 1; ivm5 <= 3; ivm5++ {
		for ivm4 = 1; ivm4 <= 5; ivm4++ {
			for ivm3 = 1; ivm3 <= 5; ivm3++ {
				for ivm2 = 1; ivm2 <= 5; ivm2++ {
					for ivm1 = 1; ivm1 <= 5; ivm1++ {
						for ndim = 1; ndim <= 6; ndim++ {

							n = idim[ndim-1]
							for i = 1; i <= n; i++ {
								for j = 1; j <= n; j++ {
									t.Set(i-1, j-1, float64(ival[i-1+(j-1+(ndim-1)*5)*5])*vm1.Get(ivm1-1))
									if i >= j {
										t.Set(i-1, j-1, t.Get(i-1, j-1)*vm5.Get(ivm5-1))
									}
								}
							}

							w = one * vm2.Get(ivm2-1)

							for i = 1; i <= n; i++ {
								b.Set(i-1, math.Cos(float64(i))*vm3.Get(ivm3-1))
							}

							for i = 1; i <= 2*n; i++ {
								d.Set(i-1, math.Sin(float64(i))*vm4.Get(ivm4-1))
							}

							norm = golapack.Dlange('1', n, n, t, work)
							k = b.Iamax(n, 1)
							normtb = norm + math.Abs(b.Get(k-1)) + math.Abs(w)

							x.Copy(n, d, 1, 1)
							knt = knt + 1
							scale, info = golapack.Dlaqtr(false, true, n, t, dum, dumm, x, work)
							if info != 0 {
								ninfo = ninfo + 1
							}

							//                       || T*x - scale*d || /
							//                         max(ulp*||T||*||x||,smlnum/ulp*||T||,smlnum)
							y.Copy(n, d, 1, 1)
							if err = y.Gemv(NoTrans, n, n, one, t, x, 1, -scale, 1); err != nil {
								panic(err)
							}
							xnorm = x.Asum(n, 1)
							resid = y.Asum(n, 1)
							domin = math.Max(smlnum, math.Max((smlnum/eps)*norm, (norm*eps)*xnorm))
							resid = resid / domin
							if resid > rmax {
								rmax = resid
								lmax = knt
							}

							x.Copy(n, d, 1, 1)
							knt = knt + 1
							scale, info = golapack.Dlaqtr(true, true, n, t, dum, dumm, x, work)
							if info != 0 {
								ninfo = ninfo + 1
							}

							//                       || T*x - scale*d || /
							//                         max(ulp*||T||*||x||,smlnum/ulp*||T||,smlnum)
							y.Copy(n, d, 1, 1)
							y.Gemv(Trans, n, n, one, t, x, 1, -scale, 1)
							xnorm = x.Asum(n, 1)
							resid = y.Asum(n, 1)
							domin = math.Max(smlnum, math.Max((smlnum/eps)*norm, (norm*eps)*xnorm))
							resid = resid / domin
							if resid > rmax {
								rmax = resid
								lmax = knt
							}

							x.Copy(2*n, d, 1, 1)
							knt = knt + 1
							scale, info = golapack.Dlaqtr(false, false, n, t, b, w, x, work)
							if info != 0 {
								ninfo = ninfo + 1
							}

							//                       ||(T+i*B)*(x1+i*x2) - scale*(d1+i*d2)|| /
							//                          max(ulp*(||T||+||B||)*(||x1||+||x2||),
							//                                  smlnum/ulp * (||T||+||B||), smlnum )
							//
							y.Copy(2*n, d, 1, 1)
							y.Set(0, x.Off(1+n-1).Dot(n, b, 1, 1)+scale*y.Get(0))
							for i = 2; i <= n; i++ {
								y.Set(i-1, w*x.Get(i+n-1)+scale*y.Get(i-1))
							}
							if err = y.Gemv(NoTrans, n, n, one, t, x, 1, -one, 1); err != nil {
								panic(err)
							}

							y.Set(1+n-1, x.Dot(n, b, 1, 1)-scale*y.Get(1+n-1))
							for i = 2; i <= n; i++ {
								y.Set(i+n-1, w*x.Get(i-1)-scale*y.Get(i+n-1))
							}
							if err = y.Off(1+n-1).Gemv(NoTrans, n, n, one, t, x.Off(1+n-1), 1, one, 1); err != nil {
								panic(err)
							}

							resid = y.Asum(2*n, 1)
							domin = math.Max(smlnum, math.Max((smlnum/eps)*normtb, eps*(normtb*x.Asum(2*n, 1))))
							resid = resid / domin
							if resid > rmax {
								rmax = resid
								lmax = knt
							}

							x.Copy(2*n, d, 1, 1)
							knt = knt + 1
							scale, info = golapack.Dlaqtr(true, false, n, t, b, w, x, work)
							if info != 0 {
								ninfo = ninfo + 1
							}

							//                       ||(T+i*B)*(x1+i*x2) - scale*(d1+i*d2)|| /
							//                          max(ulp*(||T||+||B||)*(||x1||+||x2||),
							//                                  smlnum/ulp * (||T||+||B||), smlnum )
							y.Copy(2*n, d, 1, 1)
							y.Set(0, b.Get(0)*x.Get(1+n-1)-scale*y.Get(0))
							for i = 2; i <= n; i++ {
								y.Set(i-1, b.Get(i-1)*x.Get(1+n-1)+w*x.Get(i+n-1)-scale*y.Get(i-1))
							}
							if err = y.Gemv(Trans, n, n, one, t, x, 1, one, 1); err != nil {
								panic(err)
							}

							y.Set(1+n-1, b.Get(0)*x.Get(0)+scale*y.Get(1+n-1))
							for i = 2; i <= n; i++ {
								y.Set(i+n-1, b.Get(i-1)*x.Get(0)+w*x.Get(i-1)+scale*y.Get(i+n-1))
							}
							if err = y.Off(1+n-1).Gemv(Trans, n, n, one, t, x.Off(1+n-1), 1, -one, 1); err != nil {
								panic(err)
							}

							resid = y.Asum(2*n, 1)
							domin = math.Max(smlnum, math.Max((smlnum/eps)*normtb, eps*(normtb*x.Asum(2*n, 1))))
							resid = resid / domin
							if resid > rmax {
								rmax = resid
								lmax = knt
							}

						}
					}
				}
			}
		}
	}

	return
}
