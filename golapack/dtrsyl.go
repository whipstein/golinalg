package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrsyl solves the real Sylvester matrix equation:
//
//    op(A)*X + X*op(B) = scale*C or
//    op(A)*X - X*op(B) = scale*C,
//
// where op(A) = A or A**T, and  A and B are both upper quasi-
// triangular. A is M-by-M and B is N-by-N; the right hand side C and
// the solution X are M-by-N; and scale is an output scale factor, set
// <= 1 to avoid overflow in X.
//
// A and B must be in Schur canonical form (as returned by DHSEQR), that
// is, block upper triangular with 1-by-1 and 2-by-2 diagonal blocks;
// each 2-by-2 diagonal block has its diagonal elements equal and its
// off-diagonal elements of opposite sign.
func Dtrsyl(trana, tranb byte, isgn, m, n *int, a *mat.Matrix, lda *int, b *mat.Matrix, ldb *int, c *mat.Matrix, ldc *int, scale *float64, info *int) {
	var notrna, notrnb bool
	var a11, bignum, da11, db, eps, one, scaloc, sgn, smin, smlnum, suml, sumr, xnorm, zero float64
	var ierr, j, k, k1, k2, knext, l, l1, l2, lnext int
	dum := vf(1)
	vec := mf(2, 2, opts)
	x := mf(2, 2, opts)

	zero = 0.0
	one = 1.0

	//     Decode and Test input parameters
	notrna = trana == 'N'
	notrnb = tranb == 'N'

	(*info) = 0
	if !notrna && trana != 'T' && trana != 'C' {
		(*info) = -1
	} else if !notrnb && tranb != 'T' && tranb != 'C' {
		(*info) = -2
	} else if (*isgn) != 1 && (*isgn) != -1 {
		(*info) = -3
	} else if (*m) < 0 {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < max(1, *m) {
		(*info) = -7
	} else if (*ldb) < max(1, *n) {
		(*info) = -9
	} else if (*ldc) < max(1, *m) {
		(*info) = -11
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTRSYL"), -(*info))
		return
	}

	//     Quick return if possible
	(*scale) = one
	if (*m) == 0 || (*n) == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)
	smlnum = smlnum * float64((*m)*(*n)) / eps
	bignum = one / smlnum

	smin = math.Max(smlnum, math.Max(eps*Dlange('M', m, m, a, lda, dum), eps*Dlange('M', n, n, b, ldb, dum)))

	sgn = float64(*isgn)

	if notrna && notrnb {
		//        Solve    A*X + ISGN*X*B = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        bottom-left corner column by column by
		//
		//         A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                  M                         L-1
		//        R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)].
		//                I=K+1                       J=1
		//
		//        Start column loop (index = L)
		//        L1 (L2) : column index of the first (first) row of X(K,L).
		lnext = 1
		for l = 1; l <= (*n); l++ {
			if l < lnext {
				goto label60
			}
			if l == (*n) {
				l1 = l
				l2 = l
			} else {
				if b.Get(l, l-1) != zero {
					l1 = l
					l2 = l + 1
					lnext = l + 2
				} else {
					l1 = l
					l2 = l
					lnext = l + 1
				}
			}

			//           Start row loop (index = K)
			//           K1 (K2): row index of the first (last) row of X(K,L).
			knext = (*m)
			for k = (*m); k >= 1; k-- {
				if k > knext {
					goto label50
				}
				if k == 1 {
					k1 = k
					k2 = k
				} else {
					if a.Get(k-1, k-1-1) != zero {
						k1 = k - 1
						k2 = k
						knext = k - 2
					} else {
						k1 = k
						k2 = k
						knext = k - 1
					}
				}

				if l1 == l2 && k1 == k2 {
					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))
					scaloc = one

					a11 = a.Get(k1-1, k1-1) + sgn*b.Get(l1-1, l1-1)
					da11 = math.Abs(a11)
					if da11 <= smin {
						a11 = smin
						da11 = smin
						(*info) = 1
					}
					db = math.Abs(vec.Get(0, 0))
					if da11 < one && db > one {
						if db > bignum*da11 {
							scaloc = one / db
						}
					}
					x.Set(0, 0, (vec.Get(0, 0)*scaloc)/a11)

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))

				} else if l1 == l2 && k1 != k2 {

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, a.Off(k1-1, k1-1), lda, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*b.Get(l1-1, l1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k2-1, l1-1, x.Get(1, 0))

				} else if l1 != l2 && k1 == k2 {

					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, sgn*(c.Get(k1-1, l1-1)-(suml+sgn*sumr)))

					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(1, 0, sgn*(c.Get(k1-1, l2-1)-(suml+sgn*sumr)))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, b.Off(l1-1, l1-1), ldb, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*a.Get(k1-1, k1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(1, 0))

				} else if l1 != l2 && k1 != k2 {

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(0, 1, c.Get(k1-1, l2-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(1, 1, c.Get(k2-1, l2-1)-(suml+sgn*sumr))

					Dlasy2(false, false, isgn, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a.Off(k1-1, k1-1), lda, b.Off(l1-1, l1-1), ldb, vec, func() *int { y := 2; return &y }(), &scaloc, x, func() *int { y := 2; return &y }(), &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(0, 1))
					c.Set(k2-1, l1-1, x.Get(1, 0))
					c.Set(k2-1, l2-1, x.Get(1, 1))
				}

			label50:
			}

		label60:
		}

	} else if !notrna && notrnb {
		//        Solve    A**T *X + ISGN*X*B = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        upper-left corner column by column by
		//
		//          A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                   K-1        T                    L-1
		//          R(K,L) = SUM [A(I,K)**T*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)]
		//                   I=1                          J=1
		//
		//        Start column loop (index = L)
		//        L1 (L2): column index of the first (last) row of X(K,L)
		lnext = 1
		for l = 1; l <= (*n); l++ {
			if l < lnext {
				goto label120
			}
			if l == (*n) {
				l1 = l
				l2 = l
			} else {
				if b.Get(l, l-1) != zero {
					l1 = l
					l2 = l + 1
					lnext = l + 2
				} else {
					l1 = l
					l2 = l
					lnext = l + 1
				}
			}

			//           Start row loop (index = K)
			//           K1 (K2): row index of the first (last) row of X(K,L)
			knext = 1
			for k = 1; k <= (*m); k++ {
				if k < knext {
					goto label110
				}
				if k == (*m) {
					k1 = k
					k2 = k
				} else {
					if a.Get(k, k-1) != zero {
						k1 = k
						k2 = k + 1
						knext = k + 2
					} else {
						k1 = k
						k2 = k
						knext = k + 1
					}
				}

				if l1 == l2 && k1 == k2 {
					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))
					scaloc = one

					a11 = a.Get(k1-1, k1-1) + sgn*b.Get(l1-1, l1-1)
					da11 = math.Abs(a11)
					if da11 <= smin {
						a11 = smin
						da11 = smin
						(*info) = 1
					}
					db = math.Abs(vec.Get(0, 0))
					if da11 < one && db > one {
						if db > bignum*da11 {
							scaloc = one / db
						}
					}
					x.Set(0, 0, (vec.Get(0, 0)*scaloc)/a11)

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))

				} else if l1 == l2 && k1 != k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, a.Off(k1-1, k1-1), lda, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*b.Get(l1-1, l1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k2-1, l1-1, x.Get(1, 0))

				} else if l1 != l2 && k1 == k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, sgn*(c.Get(k1-1, l1-1)-(suml+sgn*sumr)))

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(1, 0, sgn*(c.Get(k1-1, l2-1)-(suml+sgn*sumr)))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, b.Off(l1-1, l1-1), ldb, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*a.Get(k1-1, k1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(1, 0))

				} else if l1 != l2 && k1 != k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k1-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(0, 1, c.Get(k1-1, l2-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l1-1, 1))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot(l1-1, c.Vector(k2-1, 0, *ldc), b.Vector(0, l2-1, 1))
					vec.Set(1, 1, c.Get(k2-1, l2-1)-(suml+sgn*sumr))

					Dlasy2(true, false, isgn, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a.Off(k1-1, k1-1), lda, b.Off(l1-1, l1-1), ldb, vec, func() *int { y := 2; return &y }(), &scaloc, x, func() *int { y := 2; return &y }(), &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(0, 1))
					c.Set(k2-1, l1-1, x.Get(1, 0))
					c.Set(k2-1, l2-1, x.Get(1, 1))
				}

			label110:
			}
		label120:
		}

	} else if !notrna && !notrnb {
		//        Solve    A**T*X + ISGN*X*B**T = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        top-right corner column by column by
		//
		//           A(K,K)**T*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
		//
		//        Where
		//                     K-1                            N
		//            R(K,L) = SUM [A(I,K)**T*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
		//                     I=1                          J=L+1
		//
		//        Start column loop (index = L)
		//        L1 (L2): column index of the first (last) row of X(K,L)
		lnext = (*n)
		for l = (*n); l >= 1; l-- {
			if l > lnext {
				goto label180
			}
			if l == 1 {
				l1 = l
				l2 = l
			} else {
				if b.Get(l-1, l-1-1) != zero {
					l1 = l - 1
					l2 = l
					lnext = l - 2
				} else {
					l1 = l
					l2 = l
					lnext = l - 1
				}
			}

			//           Start row loop (index = K)
			//           K1 (K2): row index of the first (last) row of X(K,L)
			knext = 1
			for k = 1; k <= (*m); k++ {
				if k < knext {
					goto label170
				}
				if k == (*m) {
					k1 = k
					k2 = k
				} else {
					if a.Get(k, k-1) != zero {
						k1 = k
						k2 = k + 1
						knext = k + 2
					} else {
						k1 = k
						k2 = k
						knext = k + 1
					}
				}

				if l1 == l2 && k1 == k2 {
					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l1, c.Vector(k1-1, min(l1+1, *n)-1, *ldc), b.Vector(l1-1, min(l1+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))
					scaloc = one

					a11 = a.Get(k1-1, k1-1) + sgn*b.Get(l1-1, l1-1)
					da11 = math.Abs(a11)
					if da11 <= smin {
						a11 = smin
						da11 = smin
						(*info) = 1
					}
					db = math.Abs(vec.Get(0, 0))
					if da11 < one && db > one {
						if db > bignum*da11 {
							scaloc = one / db
						}
					}
					x.Set(0, 0, (vec.Get(0, 0)*scaloc)/a11)

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))

				} else if l1 == l2 && k1 != k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, a.Off(k1-1, k1-1), lda, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*b.Get(l1-1, l1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k2-1, l1-1, x.Get(1, 0))

				} else if l1 != l2 && k1 == k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, sgn*(c.Get(k1-1, l1-1)-(suml+sgn*sumr)))

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, sgn*(c.Get(k1-1, l2-1)-(suml+sgn*sumr)))

					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, b.Off(l1-1, l1-1), ldb, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*a.Get(k1-1, k1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(1, 0))

				} else if l1 != l2 && k1 != k2 {

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k1-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 1, c.Get(k1-1, l2-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot(k1-1, a.Vector(0, k2-1, 1), c.Vector(0, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 1, c.Get(k2-1, l2-1)-(suml+sgn*sumr))

					Dlasy2(true, true, isgn, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a.Off(k1-1, k1-1), lda, b.Off(l1-1, l1-1), ldb, vec, func() *int { y := 2; return &y }(), &scaloc, x, func() *int { y := 2; return &y }(), &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(0, 1))
					c.Set(k2-1, l1-1, x.Get(1, 0))
					c.Set(k2-1, l2-1, x.Get(1, 1))
				}

			label170:
			}
		label180:
		}

	} else if notrna && !notrnb {
		//        Solve    A*X + ISGN*X*B**T = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        bottom-right corner column by column by
		//
		//            A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L)**T = C(K,L) - R(K,L)
		//
		//        Where
		//                      M                          N
		//            R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B(L,J)**T].
		//                    I=K+1                      J=L+1
		//
		//        Start column loop (index = L)
		//        L1 (L2): column index of the first (last) row of X(K,L)
		lnext = (*n)
		for l = (*n); l >= 1; l-- {
			if l > lnext {
				goto label240
			}
			if l == 1 {
				l1 = l
				l2 = l
			} else {
				if b.Get(l-1, l-1-1) != zero {
					l1 = l - 1
					l2 = l
					lnext = l - 2
				} else {
					l1 = l
					l2 = l
					lnext = l - 1
				}
			}

			//           Start row loop (index = K)
			//           K1 (K2): row index of the first (last) row of X(K,L)
			knext = (*m)
			for k = (*m); k >= 1; k-- {
				if k > knext {
					goto label230
				}
				if k == 1 {
					k1 = k
					k2 = k
				} else {
					if a.Get(k-1, k-1-1) != zero {
						k1 = k - 1
						k2 = k
						knext = k - 2
					} else {
						k1 = k
						k2 = k
						knext = k - 1
					}
				}

				if l1 == l2 && k1 == k2 {
					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l1, c.Vector(k1-1, min(l1+1, *n)-1, *ldc), b.Vector(l1-1, min(l1+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))
					scaloc = one

					a11 = a.Get(k1-1, k1-1) + sgn*b.Get(l1-1, l1-1)
					da11 = math.Abs(a11)
					if da11 <= smin {
						a11 = smin
						da11 = smin
						(*info) = 1
					}
					db = math.Abs(vec.Get(0, 0))
					if da11 < one && db > one {
						if db > bignum*da11 {
							scaloc = one / db
						}
					}
					x.Set(0, 0, (vec.Get(0, 0)*scaloc)/a11)

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))

				} else if l1 == l2 && k1 != k2 {

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, a.Off(k1-1, k1-1), lda, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*b.Get(l1-1, l1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k2-1, l1-1, x.Get(1, 0))

				} else if l1 != l2 && k1 == k2 {

					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, sgn*(c.Get(k1-1, l1-1)-(suml+sgn*sumr)))

					suml = goblas.Ddot((*m)-k1, a.Vector(k1-1, min(k1+1, *m)-1, *lda), c.Vector(min(k1+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, sgn*(c.Get(k1-1, l2-1)-(suml+sgn*sumr)))

					Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, b.Off(l1-1, l1-1), ldb, &one, &one, vec, func() *int { y := 2; return &y }(), toPtrf64(-sgn*a.Get(k1-1, k1-1)), &zero, x, func() *int { y := 2; return &y }(), &scaloc, &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(1, 0))

				} else if l1 != l2 && k1 != k2 {

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 0, c.Get(k1-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k1-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k1-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(0, 1, c.Get(k1-1, l2-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l1-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l1-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 0, c.Get(k2-1, l1-1)-(suml+sgn*sumr))

					suml = goblas.Ddot((*m)-k2, a.Vector(k2-1, min(k2+1, *m)-1, *lda), c.Vector(min(k2+1, *m)-1, l2-1, 1))
					sumr = goblas.Ddot((*n)-l2, c.Vector(k2-1, min(l2+1, *n)-1, *ldc), b.Vector(l2-1, min(l2+1, *n)-1, *ldb))
					vec.Set(1, 1, c.Get(k2-1, l2-1)-(suml+sgn*sumr))

					Dlasy2(false, true, isgn, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), a.Off(k1-1, k1-1), lda, b.Off(l1-1, l1-1), ldb, vec, func() *int { y := 2; return &y }(), &scaloc, x, func() *int { y := 2; return &y }(), &xnorm, &ierr)
					if ierr != 0 {
						(*info) = 1
					}

					if scaloc != one {
						for j = 1; j <= (*n); j++ {
							goblas.Dscal(*m, scaloc, c.Vector(0, j-1, 1))
						}
						(*scale) = (*scale) * scaloc
					}
					c.Set(k1-1, l1-1, x.Get(0, 0))
					c.Set(k1-1, l2-1, x.Get(0, 1))
					c.Set(k2-1, l1-1, x.Get(1, 0))
					c.Set(k2-1, l2-1, x.Get(1, 1))
				}

			label230:
			}
		label240:
		}

	}
}
