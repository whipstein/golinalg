package golapack

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrsyl solves the complex Sylvester matrix equation:
//
//    op(A)*X + X*op(B) = scale*C or
//    op(A)*X - X*op(B) = scale*C,
//
// where op(A) = A or A**H, and A and B are both upper triangular. A is
// M-by-M and B is N-by-N; the right hand side C and the solution X are
// M-by-N; and scale is an output scale factor, set <= 1 to avoid
// overflow in X.
func Ztrsyl(trana, tranb mat.MatTrans, isgn, m, n int, a, b, c *mat.CMatrix) (scale float64, info int, err error) {
	var notrna, notrnb bool
	var a11, suml, sumr, vec, x11 complex128
	var bignum, da11, db, eps, one, scaloc, sgn, smin, smlnum float64
	var j, k, l int

	dum := vf(1)

	one = 1.0

	//     Decode and Test input parameters
	notrna = trana == NoTrans
	notrnb = tranb == NoTrans

	if !notrna && trana != ConjTrans {
		err = fmt.Errorf("!notrna && trana != ConjTrans: trana=%s", trana)
	} else if !notrnb && tranb != ConjTrans {
		err = fmt.Errorf("!notrnb && tranb != ConjTrans: tranb=%s", tranb)
	} else if isgn != 1 && isgn != -1 {
		err = fmt.Errorf("isgn != 1 && isgn != -1: isgn=%v", isgn)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if c.Rows < max(1, m) {
		err = fmt.Errorf("c.Rows < max(1, m): c.Rows=%v, m=%v", c.Rows, m)
	}
	if err != nil {
		gltest.Xerbla2("Ztrsyl", err)
		return
	}

	//     Quick return if possible
	scale = one
	if m == 0 || n == 0 {
		return
	}

	//     Set constants to control overflow
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)
	smlnum = smlnum * float64(m*n) / eps
	bignum = one / smlnum
	smin = math.Max(smlnum, math.Max(eps*Zlange('M', m, m, a, dum), eps*Zlange('M', n, n, b, dum)))
	sgn = float64(isgn)

	if notrna && notrnb {
		//        Solve    A*X + ISGN*X*B = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        bottom-left corner column by column by
		//
		//            A(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                    M                        L-1
		//          R(K,L) = SUM [A(K,I)*X(I,L)] +ISGN*SUM [X(K,J)*B(J,L)].
		//                  I=K+1                      J=1
		for l = 1; l <= n; l++ {
			for k = m; k >= 1; k-- {
				suml = goblas.Zdotu(m-k, a.CVector(k-1, min(k+1, m)-1), c.CVector(min(k+1, m)-1, l-1, 1))
				sumr = goblas.Zdotu(l-1, c.CVector(k-1, 0), b.CVector(0, l-1, 1))
				vec = c.Get(k-1, l-1) - (suml + complex(sgn, 0)*sumr)

				scaloc = one
				a11 = a.Get(k-1, k-1) + complex(sgn, 0)*b.Get(l-1, l-1)
				da11 = math.Abs(real(a11)) + math.Abs(imag(a11))
				if da11 <= smin {
					a11 = complex(smin, 0)
					da11 = smin
					info = 1
				}
				db = math.Abs(real(vec)) + math.Abs(imag(vec))
				if da11 < one && db > one {
					if db > bignum*da11 {
						scaloc = one / db
					}
				}
				x11 = Zladiv(vec*complex(scaloc, 0), a11)

				if scaloc != one {
					for j = 1; j <= n; j++ {
						goblas.Zdscal(m, scaloc, c.CVector(0, j-1, 1))
					}
					scale = scale * scaloc
				}
				c.Set(k-1, l-1, x11)

			}
		}

	} else if !notrna && notrnb {
		//        Solve    A**H *X + ISGN*X*B = scale*C.
		//
		//        The (K,L)th block of X is determined starting from
		//        upper-left corner column by column by
		//
		//            A**H(K,K)*X(K,L) + ISGN*X(K,L)*B(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                   K-1                           L-1
		//          R(K,L) = SUM [A**H(I,K)*X(I,L)] + ISGN*SUM [X(K,J)*B(J,L)]
		//                   I=1                           J=1
		for l = 1; l <= n; l++ {
			for k = 1; k <= m; k++ {

				suml = goblas.Zdotc(k-1, a.CVector(0, k-1, 1), c.CVector(0, l-1, 1))
				sumr = goblas.Zdotu(l-1, c.CVector(k-1, 0), b.CVector(0, l-1, 1))
				vec = c.Get(k-1, l-1) - (suml + complex(sgn, 0)*sumr)

				scaloc = one
				a11 = a.GetConj(k-1, k-1) + complex(sgn, 0)*b.Get(l-1, l-1)
				da11 = math.Abs(real(a11)) + math.Abs(imag(a11))
				if da11 <= smin {
					a11 = complex(smin, 0)
					da11 = smin
					info = 1
				}
				db = math.Abs(real(vec)) + math.Abs(imag(vec))
				if da11 < one && db > one {
					if db > bignum*da11 {
						scaloc = one / db
					}
				}

				x11 = Zladiv(vec*complex(scaloc, 0), a11)

				if scaloc != one {
					for j = 1; j <= n; j++ {
						goblas.Zdscal(m, scaloc, c.CVector(0, j-1, 1))
					}
					scale = scale * scaloc
				}
				c.Set(k-1, l-1, x11)

			}
		}

	} else if !notrna && !notrnb {
		//        Solve    A**H*X + ISGN*X*B**H = C.
		//
		//        The (K,L)th block of X is determined starting from
		//        upper-right corner column by column by
		//
		//            A**H(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                    K-1
		//           R(K,L) = SUM [A**H(I,K)*X(I,L)] +
		//                    I=1
		//                           N
		//                     ISGN*SUM [X(K,J)*B**H(L,J)].
		//                          J=L+1
		for l = n; l >= 1; l-- {
			for k = 1; k <= m; k++ {

				suml = goblas.Zdotc(k-1, a.CVector(0, k-1, 1), c.CVector(0, l-1, 1))
				sumr = goblas.Zdotc(n-l, c.CVector(k-1, min(l+1, n)-1), b.CVector(l-1, min(l+1, n)-1))
				vec = c.Get(k-1, l-1) - (suml + complex(sgn, 0)*cmplx.Conj(sumr))

				scaloc = one
				a11 = cmplx.Conj(a.Get(k-1, k-1) + complex(sgn, 0)*b.Get(l-1, l-1))
				da11 = math.Abs(real(a11)) + math.Abs(imag(a11))
				if da11 <= smin {
					a11 = complex(smin, 0)
					da11 = smin
					info = 1
				}
				db = math.Abs(real(vec)) + math.Abs(imag(vec))
				if da11 < one && db > one {
					if db > bignum*da11 {
						scaloc = one / db
					}
				}

				x11 = Zladiv(vec*complex(scaloc, 0), a11)

				if scaloc != one {
					for j = 1; j <= n; j++ {
						goblas.Zdscal(m, scaloc, c.CVector(0, j-1, 1))
					}
					scale = scale * scaloc
				}
				c.Set(k-1, l-1, x11)

			}
		}

	} else if notrna && !notrnb {
		//        Solve    A*X + ISGN*X*B**H = C.
		//
		//        The (K,L)th block of X is determined starting from
		//        bottom-left corner column by column by
		//
		//           A(K,K)*X(K,L) + ISGN*X(K,L)*B**H(L,L) = C(K,L) - R(K,L)
		//
		//        Where
		//                    M                          N
		//          R(K,L) = SUM [A(K,I)*X(I,L)] + ISGN*SUM [X(K,J)*B**H(L,J)]
		//                  I=K+1                      J=L+1
		for l = n; l >= 1; l-- {
			for k = m; k >= 1; k-- {

				suml = goblas.Zdotu(m-k, a.CVector(k-1, min(k+1, m)-1), c.CVector(min(k+1, m)-1, l-1, 1))
				sumr = goblas.Zdotc(n-l, c.CVector(k-1, min(l+1, n)-1), b.CVector(l-1, min(l+1, n)-1))
				vec = c.Get(k-1, l-1) - (suml + complex(sgn, 0)*cmplx.Conj(sumr))

				scaloc = one
				a11 = a.Get(k-1, k-1) + complex(sgn, 0)*b.GetConj(l-1, l-1)
				da11 = math.Abs(real(a11)) + math.Abs(imag(a11))
				if da11 <= smin {
					a11 = complex(smin, 0)
					da11 = smin
					info = 1
				}
				db = math.Abs(real(vec)) + math.Abs(imag(vec))
				if da11 < one && db > one {
					if db > bignum*da11 {
						scaloc = one / db
					}
				}

				x11 = Zladiv(vec*complex(scaloc, 0), a11)

				if scaloc != one {
					for j = 1; j <= n; j++ {
						goblas.Zdscal(m, scaloc, c.CVector(0, j-1, 1))
					}
					scale = scale * scaloc
				}
				c.Set(k-1, l-1, x11)

			}
		}

	}

	return
}
