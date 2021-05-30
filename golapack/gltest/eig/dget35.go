package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dget35 tests DTRSYL, a routine for solving the Sylvester matrix
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
func Dget35(rmax *float64, lmax *int, ninfo *int, knt *int) {
	var trana, tranb byte
	var bignum, cnrm, eps, four, one, res, res1, rmul, scale, smlnum, tnrm, two, xnrm, zero float64
	var i, ima, imb, imlda1, imlda2, imldb1, imloff, info, isgn, itrana, itranb, j, m, n int

	dum := vf(1)
	vm1 := vf(3)
	vm2 := vf(3)
	idim := make([]int, 8)
	a := mf(6, 6, opts)
	b := mf(6, 6, opts)
	c := mf(6, 6, opts)
	cc := mf(6, 6, opts)

	ival := make([]int, 6*6*8)

	zero = 0.0
	one = 1.0
	two = 2.0
	four = 4.0

	idim[0], idim[1], idim[2], idim[3], idim[4], idim[5], idim[6], idim[7] = 1, 2, 3, 4, 3, 3, 6, 4
	ival[0+(0+(0)*6)*6], ival[1+(0+(0)*6)*6], ival[2+(0+(0)*6)*6], ival[3+(0+(0)*6)*6], ival[4+(0+(0)*6)*6], ival[5+(0+(0)*6)*6], ival[0+(1+(0)*6)*6], ival[1+(1+(0)*6)*6], ival[2+(1+(0)*6)*6], ival[3+(1+(0)*6)*6], ival[4+(1+(0)*6)*6], ival[5+(1+(0)*6)*6], ival[0+(2+(0)*6)*6], ival[1+(2+(0)*6)*6], ival[2+(2+(0)*6)*6], ival[3+(2+(0)*6)*6], ival[4+(2+(0)*6)*6], ival[5+(2+(0)*6)*6], ival[0+(3+(0)*6)*6], ival[1+(3+(0)*6)*6], ival[2+(3+(0)*6)*6], ival[3+(3+(0)*6)*6], ival[4+(3+(0)*6)*6], ival[5+(3+(0)*6)*6], ival[0+(4+(0)*6)*6], ival[1+(4+(0)*6)*6], ival[2+(4+(0)*6)*6], ival[3+(4+(0)*6)*6], ival[4+(4+(0)*6)*6], ival[5+(4+(0)*6)*6], ival[0+(5+(0)*6)*6], ival[1+(5+(0)*6)*6], ival[2+(5+(0)*6)*6], ival[3+(5+(0)*6)*6], ival[4+(5+(0)*6)*6], ival[5+(5+(0)*6)*6], ival[0+(0+(1)*6)*6], ival[1+(0+(1)*6)*6], ival[2+(0+(1)*6)*6], ival[3+(0+(1)*6)*6], ival[4+(0+(1)*6)*6], ival[5+(0+(1)*6)*6], ival[0+(1+(1)*6)*6], ival[1+(1+(1)*6)*6], ival[2+(1+(1)*6)*6], ival[3+(1+(1)*6)*6], ival[4+(1+(1)*6)*6], ival[5+(1+(1)*6)*6], ival[0+(2+(1)*6)*6], ival[1+(2+(1)*6)*6], ival[2+(2+(1)*6)*6], ival[3+(2+(1)*6)*6], ival[4+(2+(1)*6)*6], ival[5+(2+(1)*6)*6], ival[0+(3+(1)*6)*6], ival[1+(3+(1)*6)*6], ival[2+(3+(1)*6)*6], ival[3+(3+(1)*6)*6], ival[4+(3+(1)*6)*6], ival[5+(3+(1)*6)*6], ival[0+(4+(1)*6)*6], ival[1+(4+(1)*6)*6], ival[2+(4+(1)*6)*6], ival[3+(4+(1)*6)*6], ival[4+(4+(1)*6)*6], ival[5+(4+(1)*6)*6], ival[0+(5+(1)*6)*6], ival[1+(5+(1)*6)*6], ival[2+(5+(1)*6)*6], ival[3+(5+(1)*6)*6], ival[4+(5+(1)*6)*6], ival[5+(5+(1)*6)*6], ival[0+(0+(2)*6)*6], ival[1+(0+(2)*6)*6], ival[2+(0+(2)*6)*6], ival[3+(0+(2)*6)*6], ival[4+(0+(2)*6)*6], ival[5+(0+(2)*6)*6], ival[0+(1+(2)*6)*6], ival[1+(1+(2)*6)*6], ival[2+(1+(2)*6)*6], ival[3+(1+(2)*6)*6], ival[4+(1+(2)*6)*6], ival[5+(1+(2)*6)*6], ival[0+(2+(2)*6)*6], ival[1+(2+(2)*6)*6], ival[2+(2+(2)*6)*6], ival[3+(2+(2)*6)*6], ival[4+(2+(2)*6)*6], ival[5+(2+(2)*6)*6], ival[0+(3+(2)*6)*6], ival[1+(3+(2)*6)*6], ival[2+(3+(2)*6)*6], ival[3+(3+(2)*6)*6], ival[4+(3+(2)*6)*6], ival[5+(3+(2)*6)*6], ival[0+(4+(2)*6)*6], ival[1+(4+(2)*6)*6], ival[2+(4+(2)*6)*6], ival[3+(4+(2)*6)*6], ival[4+(4+(2)*6)*6], ival[5+(4+(2)*6)*6], ival[0+(5+(2)*6)*6], ival[1+(5+(2)*6)*6], ival[2+(5+(2)*6)*6], ival[3+(5+(2)*6)*6], ival[4+(5+(2)*6)*6], ival[5+(5+(2)*6)*6], ival[0+(0+(3)*6)*6], ival[1+(0+(3)*6)*6], ival[2+(0+(3)*6)*6], ival[3+(0+(3)*6)*6], ival[4+(0+(3)*6)*6], ival[5+(0+(3)*6)*6], ival[0+(1+(3)*6)*6], ival[1+(1+(3)*6)*6], ival[2+(1+(3)*6)*6], ival[3+(1+(3)*6)*6], ival[4+(1+(3)*6)*6], ival[5+(1+(3)*6)*6], ival[0+(2+(3)*6)*6], ival[1+(2+(3)*6)*6], ival[2+(2+(3)*6)*6], ival[3+(2+(3)*6)*6], ival[4+(2+(3)*6)*6], ival[5+(2+(3)*6)*6], ival[0+(3+(3)*6)*6], ival[1+(3+(3)*6)*6], ival[2+(3+(3)*6)*6], ival[3+(3+(3)*6)*6], ival[4+(3+(3)*6)*6], ival[5+(3+(3)*6)*6], ival[0+(4+(3)*6)*6], ival[1+(4+(3)*6)*6], ival[2+(4+(3)*6)*6], ival[3+(4+(3)*6)*6], ival[4+(4+(3)*6)*6], ival[5+(4+(3)*6)*6], ival[0+(5+(3)*6)*6], ival[1+(5+(3)*6)*6], ival[2+(5+(3)*6)*6], ival[3+(5+(3)*6)*6], ival[4+(5+(3)*6)*6], ival[5+(5+(3)*6)*6], ival[0+(0+(4)*6)*6], ival[1+(0+(4)*6)*6], ival[2+(0+(4)*6)*6], ival[3+(0+(4)*6)*6], ival[4+(0+(4)*6)*6], ival[5+(0+(4)*6)*6], ival[0+(1+(4)*6)*6], ival[1+(1+(4)*6)*6], ival[2+(1+(4)*6)*6], ival[3+(1+(4)*6)*6], ival[4+(1+(4)*6)*6], ival[5+(1+(4)*6)*6], ival[0+(2+(4)*6)*6], ival[1+(2+(4)*6)*6], ival[2+(2+(4)*6)*6], ival[3+(2+(4)*6)*6], ival[4+(2+(4)*6)*6], ival[5+(2+(4)*6)*6], ival[0+(3+(4)*6)*6], ival[1+(3+(4)*6)*6], ival[2+(3+(4)*6)*6], ival[3+(3+(4)*6)*6], ival[4+(3+(4)*6)*6], ival[5+(3+(4)*6)*6], ival[0+(4+(4)*6)*6], ival[1+(4+(4)*6)*6], ival[2+(4+(4)*6)*6], ival[3+(4+(4)*6)*6], ival[4+(4+(4)*6)*6], ival[5+(4+(4)*6)*6], ival[0+(5+(4)*6)*6], ival[1+(5+(4)*6)*6], ival[2+(5+(4)*6)*6], ival[3+(5+(4)*6)*6], ival[4+(5+(4)*6)*6], ival[5+(5+(4)*6)*6], ival[0+(0+(5)*6)*6], ival[1+(0+(5)*6)*6], ival[2+(0+(5)*6)*6], ival[3+(0+(5)*6)*6], ival[4+(0+(5)*6)*6], ival[5+(0+(5)*6)*6], ival[0+(1+(5)*6)*6], ival[1+(1+(5)*6)*6], ival[2+(1+(5)*6)*6], ival[3+(1+(5)*6)*6], ival[4+(1+(5)*6)*6], ival[5+(1+(5)*6)*6], ival[0+(2+(5)*6)*6], ival[1+(2+(5)*6)*6], ival[2+(2+(5)*6)*6], ival[3+(2+(5)*6)*6], ival[4+(2+(5)*6)*6], ival[5+(2+(5)*6)*6], ival[0+(3+(5)*6)*6], ival[1+(3+(5)*6)*6], ival[2+(3+(5)*6)*6], ival[3+(3+(5)*6)*6], ival[4+(3+(5)*6)*6], ival[5+(3+(5)*6)*6], ival[0+(4+(5)*6)*6], ival[1+(4+(5)*6)*6], ival[2+(4+(5)*6)*6], ival[3+(4+(5)*6)*6], ival[4+(4+(5)*6)*6], ival[5+(4+(5)*6)*6], ival[0+(5+(5)*6)*6], ival[1+(5+(5)*6)*6], ival[2+(5+(5)*6)*6], ival[3+(5+(5)*6)*6], ival[4+(5+(5)*6)*6], ival[5+(5+(5)*6)*6], ival[0+(0+(6)*6)*6], ival[1+(0+(6)*6)*6], ival[2+(0+(6)*6)*6], ival[3+(0+(6)*6)*6], ival[4+(0+(6)*6)*6], ival[5+(0+(6)*6)*6], ival[0+(1+(6)*6)*6], ival[1+(1+(6)*6)*6], ival[2+(1+(6)*6)*6], ival[3+(1+(6)*6)*6], ival[4+(1+(6)*6)*6], ival[5+(1+(6)*6)*6], ival[0+(2+(6)*6)*6], ival[1+(2+(6)*6)*6], ival[2+(2+(6)*6)*6], ival[3+(2+(6)*6)*6], ival[4+(2+(6)*6)*6], ival[5+(2+(6)*6)*6], ival[0+(3+(6)*6)*6], ival[1+(3+(6)*6)*6], ival[2+(3+(6)*6)*6], ival[3+(3+(6)*6)*6], ival[4+(3+(6)*6)*6], ival[5+(3+(6)*6)*6], ival[0+(4+(6)*6)*6], ival[1+(4+(6)*6)*6], ival[2+(4+(6)*6)*6], ival[3+(4+(6)*6)*6], ival[4+(4+(6)*6)*6], ival[5+(4+(6)*6)*6], ival[0+(5+(6)*6)*6], ival[1+(5+(6)*6)*6], ival[2+(5+(6)*6)*6], ival[3+(5+(6)*6)*6], ival[4+(5+(6)*6)*6], ival[5+(5+(6)*6)*6], ival[0+(0+(7)*6)*6], ival[1+(0+(7)*6)*6], ival[2+(0+(7)*6)*6], ival[3+(0+(7)*6)*6], ival[4+(0+(7)*6)*6], ival[5+(0+(7)*6)*6], ival[0+(1+(7)*6)*6], ival[1+(1+(7)*6)*6], ival[2+(1+(7)*6)*6], ival[3+(1+(7)*6)*6], ival[4+(1+(7)*6)*6], ival[5+(1+(7)*6)*6], ival[0+(2+(7)*6)*6], ival[1+(2+(7)*6)*6], ival[2+(2+(7)*6)*6], ival[3+(2+(7)*6)*6], ival[4+(2+(7)*6)*6], ival[5+(2+(7)*6)*6], ival[0+(3+(7)*6)*6], ival[1+(3+(7)*6)*6], ival[2+(3+(7)*6)*6], ival[3+(3+(7)*6)*6], ival[4+(3+(7)*6)*6], ival[5+(3+(7)*6)*6], ival[0+(4+(7)*6)*6], ival[1+(4+(7)*6)*6], ival[2+(4+(7)*6)*6], ival[3+(4+(7)*6)*6], ival[4+(4+(7)*6)*6], ival[5+(4+(7)*6)*6], ival[0+(5+(7)*6)*6], ival[1+(5+(7)*6)*6], ival[2+(5+(7)*6)*6], ival[3+(5+(7)*6)*6], ival[4+(5+(7)*6)*6], ival[5+(5+(7)*6)*6] = 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 5, 1, 2, 0, 0, 0, -8, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, -5, 3, 0, 0, 0, 0, 1, 2, 1, 4, 0, 0, -3, -9, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 3, -4, 0, 0, 0, 2, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 5, 6, 3, 4, 0, 0, -1, -9, -5, 2, 0, 0, 8, 8, 8, 8, 5, 6, 9, 9, 9, 9, -7, 5, 1, 0, 0, 0, 0, 0, 1, 5, 2, 0, 0, 0, 2, -21, 5, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

	//     Get machine parameters
	eps = golapack.Dlamch(Precision)
	smlnum = golapack.Dlamch(SafeMinimum) * four / eps
	bignum = one / smlnum
	golapack.Dlabad(&smlnum, &bignum)

	//     Set up test case parameters
	vm1.Set(0, math.Sqrt(smlnum))
	vm1.Set(1, one)
	vm1.Set(2, math.Sqrt(bignum))
	vm2.Set(0, one)
	vm2.Set(1, one+two*eps)
	vm2.Set(2, two)

	(*knt) = 0
	(*ninfo) = 0
	(*lmax) = 0
	(*rmax) = zero

	//     Begin test loop
	for itrana = 1; itrana <= 2; itrana++ {
		for itranb = 1; itranb <= 2; itranb++ {
			for isgn = -1; isgn <= 1; isgn += 2 {
				for ima = 1; ima <= 8; ima++ {
					for imlda1 = 1; imlda1 <= 3; imlda1++ {
						for imlda2 = 1; imlda2 <= 3; imlda2++ {
							for imloff = 1; imloff <= 2; imloff++ {
								for imb = 1; imb <= 8; imb++ {
									for imldb1 = 1; imldb1 <= 3; imldb1++ {
										if itrana == 1 {
											trana = 'N'
										}
										if itrana == 2 {
											trana = 'T'
										}
										if itranb == 1 {
											tranb = 'N'
										}
										if itranb == 2 {
											tranb = 'T'
										}
										m = idim[ima-1]
										n = idim[imb-1]
										tnrm = zero
										for i = 1; i <= m; i++ {
											for j = 1; j <= m; j++ {
												a.Set(i-1, j-1, float64(ival[i-1+(j-1+(ima-1)*6)*6]))
												if int(math.Abs(float64(i-j))) <= 1 {
													a.Set(i-1, j-1, a.Get(i-1, j-1)*vm1.Get(imlda1-1))
													a.Set(i-1, j-1, a.Get(i-1, j-1)*vm2.Get(imlda2-1))
												} else {
													a.Set(i-1, j-1, a.Get(i-1, j-1)*vm1.Get(imloff-1))
												}
												tnrm = maxf64(tnrm, math.Abs(a.Get(i-1, j-1)))
											}
										}
										for i = 1; i <= n; i++ {
											for j = 1; j <= n; j++ {
												b.Set(i-1, j-1, float64(ival[i-1+(j-1+(imb-1)*6)*6]))
												if int(math.Abs(float64(i-j))) <= 1 {
													b.Set(i-1, j-1, b.Get(i-1, j-1)*vm1.Get(imldb1-1))
												} else {
													b.Set(i-1, j-1, b.Get(i-1, j-1)*vm1.Get(imloff-1))
												}
												tnrm = maxf64(tnrm, math.Abs(b.Get(i-1, j-1)))
											}
										}
										cnrm = zero
										for i = 1; i <= m; i++ {
											for j = 1; j <= n; j++ {
												c.Set(i-1, j-1, math.Sin(float64(i*j)))
												cnrm = maxf64(cnrm, c.Get(i-1, j-1))
												cc.Set(i-1, j-1, c.Get(i-1, j-1))
											}
										}
										(*knt) = (*knt) + 1
										golapack.Dtrsyl(trana, tranb, &isgn, &m, &n, a, func() *int { y := 6; return &y }(), b, func() *int { y := 6; return &y }(), c, func() *int { y := 6; return &y }(), &scale, &info)
										if info != 0 {
											(*ninfo) = (*ninfo) + 1
										}
										xnrm = golapack.Dlange('M', &m, &n, c, func() *int { y := 6; return &y }(), dum)
										rmul = one
										if xnrm > one && tnrm > one {
											if xnrm > bignum/tnrm {
												rmul = one / maxf64(xnrm, tnrm)
											}
										}
										goblas.Dgemm(mat.TransByte(trana), NoTrans, &m, &n, &m, &rmul, a, func() *int { y := 6; return &y }(), c, func() *int { y := 6; return &y }(), toPtrf64(-scale*rmul), cc, func() *int { y := 6; return &y }())
										goblas.Dgemm(NoTrans, mat.TransByte(tranb), &m, &n, &n, toPtrf64(float64(isgn)*rmul), c, func() *int { y := 6; return &y }(), b, func() *int { y := 6; return &y }(), &one, cc, func() *int { y := 6; return &y }())
										res1 = golapack.Dlange('M', &m, &n, cc, func() *int { y := 6; return &y }(), dum)
										res = res1 / maxf64(smlnum, smlnum*xnrm, ((rmul*tnrm)*eps)*xnrm)
										if res > (*rmax) {
											(*lmax) = (*knt)
											(*rmax) = res
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
