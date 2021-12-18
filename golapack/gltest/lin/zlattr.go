package lin

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// lattr generates a triangular test matrix in 2-dimensional storage.
// IMAT and UPLO uniquely specify the properties of the test matrix,
// which is returned in the array A.
func zlattr(imat int, uplo mat.MatUplo, trans mat.MatTrans, iseed *[]int, n int, a *mat.CMatrix, b, work *mat.CVector, rwork *mat.Vector) (diag mat.MatDiag) {
	var upper bool
	var dist, _type byte
	var plus1, plus2, ra, rb, s, star1 complex128
	var anorm, bignum, bnorm, bscal, c, cndnum, one, rexp, sfac, smlnum, texp, tleft, tscal, two, ulp, unfl, x, y, z, zero float64
	var i, iy, j, jcount, kl, ku, mode int
	var err error

	one = 1.0
	two = 2.0
	zero = 0.0

	path := "Ztr"
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	smlnum = unfl
	bignum = (one - ulp) / smlnum
	smlnum, bignum = golapack.Dlabad(smlnum, bignum)
	if (imat >= 7 && imat <= 10) || imat == 18 {
		diag = Unit
	} else {
		diag = NonUnit
	}

	//     Quick return if N.LE.0.
	if n <= 0 {
		return
	}

	//     Call ZLATB4 to set parameters for CLATMS.
	upper = uplo == Upper
	if upper {
		_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, imat, n, n)
	} else {
		_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, -imat, n, n)
	}

	//     IMAT <= 6:  Non-unit triangular matrix
	if imat <= 6 {
		if err = matgen.Zlatms(n, n, dist, iseed, _type, rwork, mode, cndnum, anorm, kl, ku, 'N', a, work); err != nil {
			panic(err)
		}

		//     IMAT > 6:  Unit triangular matrix
		//     The diagonal is deliberately set to something other than 1.
		//
		//     IMAT = 7:  Matrix is the identity
	} else if imat == 7 {
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					a.SetRe(i-1, j-1, zero)
				}
				a.SetRe(j-1, j-1, float64(j))
			}
		} else {
			for j = 1; j <= n; j++ {
				a.SetRe(j-1, j-1, float64(j))
				for i = j + 1; i <= n; i++ {
					a.SetRe(i-1, j-1, zero)
				}
			}
		}

		//     IMAT > 7:  Non-trivial unit triangular matrix
		//
		//     Generate a unit triangular matrix T with condition CNDNUM by
		//     forming a triangular matrix with known singular values and
		//     filling in the zero entries with Givens rotations.
	} else if imat <= 10 {
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					a.SetRe(i-1, j-1, zero)
				}
				a.SetRe(j-1, j-1, float64(j))
			}
		} else {
			for j = 1; j <= n; j++ {
				a.SetRe(j-1, j-1, float64(j))
				for i = j + 1; i <= n; i++ {
					a.SetRe(i-1, j-1, zero)
				}
			}
		}

		//        Since the trace of a unit triangular matrix is 1, the product
		//        of its singular values must be 1.  Let s = sqrt(CNDNUM),
		//        x = sqrt(s) - 1/sqrt(s), y = sqrt(2/(n-2))*x, and z = x**2.
		//        The following triangular matrix has singular values s, 1, 1,
		//        ..., 1, 1/s:
		//
		//        1  y  y  y  ...  y  y  z
		//           1  0  0  ...  0  0  y
		//              1  0  ...  0  0  y
		//                 .  ...  .  .  .
		//                     .   .  .  .
		//                         1  0  y
		//                            1  y
		//                               1
		//
		//        To fill in the zeros, we first multiply by a matrix with small
		//        condition number of the form
		//
		//        1  0  0  0  0  ...
		//           1  +  *  0  0  ...
		//              1  +  0  0  0
		//                 1  +  *  0  0
		//                    1  +  0  0
		//                       ...
		//                          1  +  0
		//                             1  0
		//                                1
		//
		//        Each element marked with a '*' is formed by taking the product
		//        of the adjacent elements marked with '+'.  The '*'s can be
		//        chosen freely, and the '+'s are chosen so that the inverse of
		//        T will have elements of the same magnitude as T.  If the *'s in
		//        both T and inv(T) have small magnitude, T is well conditioned.
		//        The two offdiagonals of T are stored in WORK.
		//
		//        The product of these two matrices has the form
		//
		//        1  y  y  y  y  y  .  y  y  z
		//           1  +  *  0  0  .  0  0  y
		//              1  +  0  0  .  0  0  y
		//                 1  +  *  .  .  .  .
		//                    1  +  .  .  .  .
		//                       .  .  .  .  .
		//                          .  .  .  .
		//                             1  +  y
		//                                1  y
		//                                   1
		//
		//        Now we multiply by Givens rotations, using the fact that
		//
		//              [  c   s ] [  1   w ] [ -c  -s ] =  [  1  -w ]
		//              [ -s   c ] [  0   1 ] [  s  -c ]    [  0   1 ]
		//        and
		//              [ -c  -s ] [  1   0 ] [  c   s ] =  [  1   0 ]
		//              [  s  -c ] [  w   1 ] [ -s   c ]    [ -w   1 ]
		//
		//        where c = w / sqrt(w**2+4) and s = 2 / sqrt(w**2+4).
		star1 = 0.25 * matgen.Zlarnd(5, *iseed)
		sfac = 0.5
		plus1 = complex(sfac, 0) * matgen.Zlarnd(5, *iseed)
		for j = 1; j <= n; j += 2 {
			plus2 = star1 / plus1
			work.Set(j-1, plus1)
			work.Set(n+j-1, star1)
			if j+1 <= n {
				work.Set(j, plus2)
				work.SetRe(n+j, zero)
				plus1 = star1 / plus2
				rexp = matgen.Dlarnd(2, iseed)
				if rexp < zero {
					star1 = complex(-math.Pow(sfac, one-rexp), 0) * matgen.Zlarnd(5, *iseed)
				} else {
					star1 = complex(math.Pow(sfac, one+rexp), 0) * matgen.Zlarnd(5, *iseed)
				}
			}
		}

		x = math.Sqrt(cndnum) - 1/math.Sqrt(cndnum)
		if n > 2 {
			y = math.Sqrt(2./float64(n-2)) * x
		} else {
			y = zero
		}
		z = x * x

		if upper {
			if n > 3 {
				a.Off(1, 2).CVector().Copy(n-3, work, 1, a.Rows+1)
				if n > 4 {
					a.Off(1, 3).CVector().Copy(n-4, work.Off(n), 1, a.Rows+1)
				}
			}
			for j = 2; j <= n-1; j++ {
				a.SetRe(0, j-1, y)
				a.SetRe(j-1, n-1, y)
			}
			a.SetRe(0, n-1, z)
		} else {
			if n > 3 {
				a.Off(2, 1).CVector().Copy(n-3, work, 1, a.Rows+1)
				if n > 4 {
					a.Off(3, 1).CVector().Copy(n-4, work.Off(n), 1, a.Rows+1)
				}
			}
			for j = 2; j <= n-1; j++ {
				a.SetRe(j-1, 0, y)
				a.SetRe(n-1, j-1, y)
			}
			a.SetRe(n-1, 0, z)
		}

		//        Fill in the zeros using Givens rotations.
		if upper {
			for j = 1; j <= n-1; j++ {
				ra = a.Get(j-1, j)
				rb = 2.0
				c, s, ra = goblas.Zrotg(ra, rb, c, s)

				//              Multiply by [ c  s; -conjg(s)  c] on the left.
				if n > j+1 {
					golapack.Zrot(n-j-1, a.Off(j-1, j+2-1).CVector(), a.Rows, a.Off(j, j+2-1).CVector(), a.Rows, c, s)
				}

				//              Multiply by [-c -s;  conjg(s) -c] on the right.
				if j > 1 {
					golapack.Zrot(j-1, a.Off(0, j).CVector(), 1, a.Off(0, j-1).CVector(), 1, -c, -s)
				}

				//              Negate A(J,J+1).
				a.Set(j-1, j, -a.Get(j-1, j))
			}
		} else {
			for j = 1; j <= n-1; j++ {
				ra = a.Get(j, j-1)
				rb = 2.0
				c, s, ra = goblas.Zrotg(ra, rb, c, s)
				s = cmplx.Conj(s)

				//              Multiply by [ c -s;  conjg(s) c] on the right.
				if n > j+1 {
					golapack.Zrot(n-j-1, a.Off(j+2-1, j).CVector(), 1, a.Off(j+2-1, j-1).CVector(), 1, c, -s)
				}

				//              Multiply by [-c  s; -conjg(s) -c] on the left.
				if j > 1 {
					golapack.Zrot(j-1, a.Off(j-1, 0).CVector(), a.Rows, a.Off(j, 0).CVector(), a.Rows, -c, s)
				}

				//              Negate A(J+1,J).
				a.Set(j, j-1, -a.Get(j, j-1))
			}
		}

		//     IMAT > 10:  Pathological test cases.  These triangular matrices
		//     are badly scaled or badly conditioned, so when used in solving a
		//     triangular system they may cause overflow in the solution vector.
	} else if imat == 11 {
		//        Type 11:  Generate a triangular matrix with elements between
		//        -1 and 1. Give the diagonal norm 2 to make it well-conditioned.
		//        Make the right hand side large so that it requires scaling.
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, a.Off(0, j-1).CVector())
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
			}
		} else {
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, a.Off(j, j-1).CVector())
				}
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(2, iseed, n, b)
		iy = b.Iamax(n, 1)
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		b.Dscal(n, bscal, 1)

	} else if imat == 12 {
		//        Type 12:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 12, the offdiagonal elements are small (CNORM(j) < 1).
		golapack.Zlarnv(2, iseed, n, b)
		tscal = one / math.Max(one, float64(n-1))
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, a.Off(0, j-1).CVector())
				a.Off(0, j-1).CVector().Dscal(j-1, tscal, 1)
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
			}
			a.Set(n-1, n-1, complex(smlnum, 0)*a.Get(n-1, n-1))
		} else {
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, a.Off(j, j-1).CVector())
					a.Off(j, j-1).CVector().Dscal(n-j, tscal, 1)
				}
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
			}
			a.Set(0, 0, complex(smlnum, 0)*a.Get(0, 0))
		}

	} else if imat == 13 {
		//        Type 13:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 13, the offdiagonal elements are O(1) (CNORM(j) > 1).
		golapack.Zlarnv(2, iseed, n, b)
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, a.Off(0, j-1).CVector())
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
			}
			a.Set(n-1, n-1, complex(smlnum, 0)*a.Get(n-1, n-1))
		} else {
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, a.Off(j, j-1).CVector())
				}
				a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
			}
			a.Set(0, 0, complex(smlnum, 0)*a.Get(0, 0))
		}

	} else if imat == 14 {
		//        Type 14:  T is diagonal with small numbers on the diagonal to
		//        make the growth factor underflow, but a small right hand side
		//        chosen so that the solution does not overflow.
		if upper {
			jcount = 1
			for j = n; j >= 1; j -= 1 {
				for i = 1; i <= j-1; i++ {
					a.SetRe(i-1, j-1, zero)
				}
				if jcount <= 2 {
					a.Set(j-1, j-1, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
				} else {
					a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
				}
				jcount = jcount + 1
				if jcount > 4 {
					jcount = 1
				}
			}
		} else {
			jcount = 1
			for j = 1; j <= n; j++ {
				for i = j + 1; i <= n; i++ {
					a.SetRe(i-1, j-1, zero)
				}
				if jcount <= 2 {
					a.Set(j-1, j-1, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
				} else {
					a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed))
				}
				jcount = jcount + 1
				if jcount > 4 {
					jcount = 1
				}
			}
		}

		//        Set the right hand side alternately zero and small.
		if upper {
			b.SetRe(0, zero)
			for i = n; i >= 2; i -= 2 {
				b.SetRe(i-1, zero)
				b.Set(i-1-1, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
			}
		} else {
			b.SetRe(n-1, zero)
			for i = 1; i <= n-1; i += 2 {
				b.SetRe(i-1, zero)
				b.Set(i, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
			}
		}

	} else if imat == 15 {
		//        Type 15:  Make the diagonal elements small to cause gradual
		//        overflow when dividing by T(j,j).  To control the amount of
		//        scaling needed, the matrix is bidiagonal.
		texp = one / math.Max(one, float64(n-1))
		tscal = math.Pow(smlnum, texp)
		golapack.Zlarnv(4, iseed, n, b)
		if upper {
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-2; i++ {
					a.Set(i-1, j-1, 0.)
				}
				if j > 1 {
					a.Set(j-1-1, j-1, complex(-one, -one))
				}
				a.Set(j-1, j-1, complex(tscal, 0)*matgen.Zlarnd(5, *iseed))
			}
			b.Set(n-1, complex(one, one))
		} else {
			for j = 1; j <= n; j++ {
				for i = j + 2; i <= n; i++ {
					a.Set(i-1, j-1, 0.)
				}
				if j < n {
					a.Set(j, j-1, complex(-one, -one))
				}
				a.Set(j-1, j-1, complex(tscal, 0)*matgen.Zlarnd(5, *iseed))
			}
			b.Set(0, complex(one, one))
		}

	} else if imat == 16 {
		//        Type 16:  One zero diagonal element.
		iy = n/2 + 1
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, a.Off(0, j-1).CVector())
				if j != iy {
					a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				} else {
					a.SetRe(j-1, j-1, zero)
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, a.Off(j, j-1).CVector())
				}
				if j != iy {
					a.Set(j-1, j-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				} else {
					a.SetRe(j-1, j-1, zero)
				}
			}
		}
		golapack.Zlarnv(2, iseed, n, b)
		b.Dscal(n, two, 1)

	} else if imat == 17 {
		//        Type 17:  Make the offdiagonal elements large to cause overflow
		//        when adding a column of T.  In the non-transposed case, the
		//        matrix is constructed to cause overflow when adding a column in
		//        every other step.
		tscal = unfl / ulp
		tscal = (one - ulp) / tscal
		for j = 1; j <= n; j++ {
			for i = 1; i <= n; i++ {
				a.Set(i-1, j-1, 0.)
			}
		}
		texp = one
		if upper {
			for j = n; j >= 2; j -= 2 {
				a.SetRe(0, j-1, -tscal/float64(n+1))
				a.SetRe(j-1, j-1, one)
				b.SetRe(j-1, texp*(one-ulp))
				a.SetRe(0, j-1-1, -(tscal/float64(n+1))/float64(n+2))
				a.SetRe(j-1-1, j-1-1, one)
				b.SetRe(j-1-1, texp*float64(n*n+n-1))
				texp = texp * 2.
			}
			b.SetRe(0, (float64(n+1)/float64(n+2))*tscal)
		} else {
			for j = 1; j <= n-1; j += 2 {
				a.SetRe(n-1, j-1, -tscal/float64(n+1))
				a.SetRe(j-1, j-1, one)
				b.SetRe(j-1, texp*(one-ulp))
				a.SetRe(n-1, j, -(tscal/float64(n+1))/float64(n+2))
				a.SetRe(j, j, one)
				b.SetRe(j, texp*float64(n*n+n-1))
				texp = texp * 2.
			}
			b.SetRe(n-1, (float64(n+1)/float64(n+2))*tscal)
		}

	} else if imat == 18 {
		//        Type 18:  Generate a unit triangular matrix with elements
		//        between -1 and 1, and make the right hand side large so that it
		//        requires scaling.
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, a.Off(0, j-1).CVector())
				a.SetRe(j-1, j-1, zero)
			}
		} else {
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, a.Off(j, j-1).CVector())
				}
				a.SetRe(j-1, j-1, zero)
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(2, iseed, n, b)
		iy = b.Iamax(n, 1)
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		b.Dscal(n, bscal, 1)

	} else if imat == 19 {
		//        Type 19:  Generate a triangular matrix with elements between
		//        BIGNUM/(n-1) and BIGNUM so that at least one of the column
		//        norms will exceed BIGNUM.
		//        1/3/91:  ZLATRS no longer can handle this case
		tleft = bignum / math.Max(one, float64(n-1))
		tscal = bignum * (float64(n-1) / math.Max(one, float64(n)))
		if upper {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(5, iseed, j, a.Off(0, j-1).CVector())
				golapack.Dlarnv(1, iseed, j, rwork)
				for i = 1; i <= j; i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(tleft+rwork.Get(i-1)*tscal, 0))
				}
			}
		} else {
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(5, iseed, n-j+1, a.Off(j-1, j-1).CVector())
				golapack.Dlarnv(1, iseed, n-j+1, rwork)
				for i = j; i <= n; i++ {
					a.Set(i-1, j-1, a.Get(i-1, j-1)*complex(tleft+rwork.Get(i-j)*tscal, 0))
				}
			}
		}
		golapack.Zlarnv(2, iseed, n, b)
		b.Dscal(n, two, 1)
	}

	//     Flip the matrix if the transpose will be used.
	if trans != NoTrans {
		if upper {
			for j = 1; j <= n/2; j++ {
				a.Off(j, n-j).CVector().Swap(n-2*j+1, a.Off(j-1, j-1).CVector(), a.Rows, -1)
			}
		} else {
			for j = 1; j <= n/2; j++ {
				a.Off(n-j, j).CVector().Swap(n-2*j+1, a.Off(j-1, j-1).CVector(), 1, -a.Rows)
			}
		}
	}

	return
}
