package lin

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// zlattp generates a triangular test matrix in packed storage.
// IMAT and UPLO uniquely specify the properties of the test matrix,
// which is returned in the array AP.
func zlattp(imat int, uplo mat.MatUplo, trans mat.MatTrans, iseed *[]int, n int, ap, b, work *mat.CVector, rwork *mat.Vector) (diag mat.MatDiag) {
	var upper bool
	var dist, packit, _type byte
	var ctemp, plus1, plus2, ra, rb, s, star1 complex128
	var anorm, bignum, bnorm, bscal, c, cndnum, one, rexp, sfac, smlnum, t, texp, tleft, tscal, two, ulp, unfl, x, y, z, zero float64
	var i, iy, j, jc, jcnext, jcount, jj, jl, jr, jx, kl, ku, mode int
	var err error

	one = 1.0
	two = 2.0
	zero = 0.0

	path := "Ztp"
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
		packit = 'C'
	} else {
		_type, kl, ku, anorm, mode, cndnum, dist = zlatb4(path, -imat, n, n)
		packit = 'R'
	}

	//     IMAT <= 6:  Non-unit triangular matrix
	if imat <= 6 {
		if err = matgen.Zlatms(n, n, dist, iseed, _type, rwork, mode, cndnum, anorm, kl, ku, packit, ap.CMatrix(n, opts), work); err != nil {
			panic(err)
		}

		//     IMAT > 6:  Unit triangular matrix
		//     The diagonal is deliberately set to something other than 1.
		//
		//     IMAT = 7:  Matrix is the identity
	} else if imat == 7 {
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					ap.SetRe(jc+i-1-1, zero)
				}
				ap.SetRe(jc+j-1-1, float64(j))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				ap.SetRe(jc-1, float64(j))
				for i = j + 1; i <= n; i++ {
					ap.SetRe(jc+i-j-1, zero)
				}
				jc = jc + n - j + 1
			}
		}

		//     IMAT > 7:  Non-trivial unit triangular matrix
		//
		//     Generate a unit triangular matrix T with condition CNDNUM by
		//     forming a triangular matrix with known singular values and
		//     filling in the zero entries with Givens rotations.
	} else if imat <= 10 {
		if upper {
			jc = 0
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					ap.SetRe(jc+i-1, zero)
				}
				ap.SetRe(jc+j-1, float64(j))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				ap.SetRe(jc-1, float64(j))
				for i = j + 1; i <= n; i++ {
					ap.SetRe(jc+i-j-1, zero)
				}
				jc = jc + n - j + 1
			}
		}

		//        Since the trace of a unit triangular matrix is 1, the product
		//        of its singular values must be 1.  Let s = math.Sqrt(CNDNUM),
		//        x = math.Sqrt(s) - 1/math.Sqrt(s), y = math.Sqrt(2/(n-2))*x, and z = x**2.
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
		//        where c = w / math.Sqrt(w**2+4) and s = 2 / math.Sqrt(w**2+4).
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
				rexp = real(matgen.Zlarnd(2, *iseed))
				if rexp < zero {
					star1 = complex(-math.Pow(sfac, one-rexp), 0) * matgen.Zlarnd(5, *iseed)
				} else {
					star1 = complex(math.Pow(sfac, one+rexp), 0) * matgen.Zlarnd(5, *iseed)
				}
			}
		}

		x = math.Sqrt(cndnum) - one/math.Sqrt(cndnum)
		if n > 2 {
			y = math.Sqrt(two/float64(n-2)) * x
		} else {
			y = zero
		}
		z = x * x

		if upper {
			//           Set the upper triangle of A with a unit triangular matrix
			//           of known condition number.
			jc = 1
			for j = 2; j <= n; j++ {
				ap.SetRe(jc, y)
				if j > 2 {
					ap.Set(jc+j-1-1, work.Get(j-2-1))
				}
				if j > 3 {
					ap.Set(jc+j-2-1, work.Get(n+j-3-1))
				}
				jc = jc + j
			}
			jc = jc - n
			ap.SetRe(jc, z)
			for j = 2; j <= n-1; j++ {
				ap.SetRe(jc+j-1, y)
			}
		} else {
			//           Set the lower triangle of A with a unit triangular matrix
			//           of known condition number.
			for i = 2; i <= n-1; i++ {
				ap.SetRe(i-1, y)
			}
			ap.SetRe(n-1, z)
			jc = n + 1
			for j = 2; j <= n-1; j++ {
				ap.Set(jc, work.Get(j-1-1))
				if j < n-1 {
					ap.Set(jc+2-1, work.Get(n+j-1-1))
				}
				ap.SetRe(jc+n-j-1, y)
				jc = jc + n - j + 1
			}
		}

		//        Fill in the zeros using Givens rotations
		if upper {
			jc = 1
			for j = 1; j <= n-1; j++ {
				jcnext = jc + j
				ra = ap.Get(jcnext + j - 1 - 1)
				rb = complex(two, 0)
				c, s, ra = goblas.Zrotg(ra, rb, c, s)

				//              Multiply by [ c  s; -conjg(s)  c] on the left.
				if n > j+1 {
					jx = jcnext + j
					for i = j + 2; i <= n; i++ {
						ctemp = complex(c, 0)*ap.Get(jx+j-1) + s*ap.Get(jx+j)
						ap.Set(jx+j, -cmplx.Conj(s)*ap.Get(jx+j-1)+complex(c, 0)*ap.Get(jx+j))
						ap.Set(jx+j-1, ctemp)
						jx = jx + i
					}
				}

				//              Multiply by [-c -s;  conjg(s) -c] on the right.
				if j > 1 {
					golapack.Zrot(j-1, ap.Off(jcnext-1, 1), ap.Off(jc-1, 1), -c, -s)
				}

				//              Negate A(J,J+1).
				ap.Set(jcnext+j-1-1, -ap.Get(jcnext+j-1-1))
				jc = jcnext
			}
		} else {
			jc = 1
			for j = 1; j <= n-1; j++ {
				jcnext = jc + n - j + 1
				ra = ap.Get(jc + 1 - 1)
				rb = complex(two, 0)
				c, s, ra = goblas.Zrotg(ra, rb, c, s)
				s = cmplx.Conj(s)

				//              Multiply by [ c -s;  conjg(s) c] on the right.
				if n > j+1 {
					golapack.Zrot(n-j-1, ap.Off(jcnext, 1), ap.Off(jc+2-1, 1), c, -s)
				}

				//              Multiply by [-c  s; -conjg(s) -c] on the left.
				if j > 1 {
					jx = 1
					for i = 1; i <= j-1; i++ {
						ctemp = complex(-c, 0)*ap.Get(jx+j-i-1) + s*ap.Get(jx+j-i)
						ap.Set(jx+j-i, -cmplx.Conj(s)*ap.Get(jx+j-i-1)-complex(c, 0)*ap.Get(jx+j-i))
						ap.Set(jx+j-i-1, ctemp)
						jx = jx + n - i + 1
					}
				}

				//              Negate A(J+1,J).
				ap.Set(jc, -ap.Get(jc))
				jc = jcnext
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
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, ap.Off(jc-1))
				ap.Set(jc+j-1-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, ap.Off(jc))
				}
				ap.Set(jc-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				jc = jc + n - j + 1
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(2, iseed, n, b)
		iy = goblas.Izamax(n, b.Off(0, 1))
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		goblas.Zdscal(n, bscal, b.Off(0, 1))

	} else if imat == 12 {
		//        Type 12:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 12, the offdiagonal elements are small (CNORM(j) < 1).
		golapack.Zlarnv(2, iseed, n, b)
		tscal = one / math.Max(one, float64(n-1))
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, ap.Off(jc-1))
				goblas.Zdscal(j-1, tscal, ap.Off(jc-1, 1))
				ap.Set(jc+j-1-1, matgen.Zlarnd(5, *iseed))
				jc = jc + j
			}
			ap.Set(n*(n+1)/2-1, complex(smlnum, 0)*ap.Get(n*(n+1)/2-1))
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(2, iseed, n-j, ap.Off(jc))
				goblas.Zdscal(n-j, tscal, ap.Off(jc, 1))
				ap.Set(jc-1, matgen.Zlarnd(5, *iseed))
				jc = jc + n - j + 1
			}
			ap.Set(0, complex(smlnum, 0)*ap.Get(0))
		}

	} else if imat == 13 {
		//        Type 13:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 13, the offdiagonal elements are O(1) (CNORM(j) > 1).
		golapack.Zlarnv(2, iseed, n, b)
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, ap.Off(jc-1))
				ap.Set(jc+j-1-1, matgen.Zlarnd(5, *iseed))
				jc = jc + j
			}
			ap.Set(n*(n+1)/2-1, complex(smlnum, 0)*ap.Get(n*(n+1)/2-1))
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, n-j, ap.Off(jc))
				ap.Set(jc-1, matgen.Zlarnd(5, *iseed))
				jc = jc + n - j + 1
			}
			ap.Set(0, complex(smlnum, 0)*ap.Get(0))
		}

	} else if imat == 14 {
		//        Type 14:  T is diagonal with small numbers on the diagonal to
		//        make the growth factor underflow, but a small right hand side
		//        chosen so that the solution does not overflow.
		if upper {
			jcount = 1
			jc = (n-1)*n/2 + 1
			for j = n; j >= 1; j -= 1 {
				for i = 1; i <= j-1; i++ {
					ap.SetRe(jc+i-1-1, zero)
				}
				if jcount <= 2 {
					ap.Set(jc+j-1-1, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
				} else {
					ap.Set(jc+j-1-1, matgen.Zlarnd(5, *iseed))
				}
				jcount = jcount + 1
				if jcount > 4 {
					jcount = 1
				}
				jc = jc - j + 1
			}
		} else {
			jcount = 1
			jc = 1
			for j = 1; j <= n; j++ {
				for i = j + 1; i <= n; i++ {
					ap.SetRe(jc+i-j-1, zero)
				}
				if jcount <= 2 {
					ap.Set(jc-1, complex(smlnum, 0)*matgen.Zlarnd(5, *iseed))
				} else {
					ap.Set(jc-1, matgen.Zlarnd(5, *iseed))
				}
				jcount = jcount + 1
				if jcount > 4 {
					jcount = 1
				}
				jc = jc + n - j + 1
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
			jc = 1
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-2; i++ {
					ap.SetRe(jc+i-1-1, zero)
				}
				if j > 1 {
					ap.Set(jc+j-2-1, complex(-one, -one))
				}
				ap.Set(jc+j-1-1, complex(tscal, 0)*matgen.Zlarnd(5, *iseed))
				jc = jc + j
			}
			b.Set(n-1, complex(one, one))
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				for i = j + 2; i <= n; i++ {
					ap.SetRe(jc+i-j-1, zero)
				}
				if j < n {
					ap.Set(jc, complex(-one, -one))
				}
				ap.Set(jc-1, complex(tscal, 0)*matgen.Zlarnd(5, *iseed))
				jc = jc + n - j + 1
			}
			b.Set(0, complex(one, one))
		}

	} else if imat == 16 {
		//        Type 16:  One zero diagonal element.
		iy = n/2 + 1
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j, ap.Off(jc-1))
				if j != iy {
					ap.Set(jc+j-1-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				} else {
					ap.SetRe(jc+j-1-1, zero)
				}
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, n-j+1, ap.Off(jc-1))
				if j != iy {
					ap.Set(jc-1, matgen.Zlarnd(5, *iseed)*complex(two, 0))
				} else {
					ap.SetRe(jc-1, zero)
				}
				jc = jc + n - j + 1
			}
		}
		golapack.Zlarnv(2, iseed, n, b)
		goblas.Zdscal(n, two, b.Off(0, 1))

	} else if imat == 17 {
		//        Type 17:  Make the offdiagonal elements large to cause overflow
		//        when adding a column of T.  In the non-transposed case, the
		//        matrix is constructed to cause overflow when adding a column in
		//        every other step.
		tscal = unfl / ulp
		tscal = (one - ulp) / tscal
		for j = 1; j <= n*(n+1)/2; j++ {
			ap.SetRe(j-1, zero)
		}
		texp = one
		if upper {
			jc = (n-1)*n/2 + 1
			for j = n; j >= 2; j -= 2 {
				ap.SetRe(jc-1, -tscal/float64(n+1))
				ap.SetRe(jc+j-1-1, one)
				b.SetRe(j-1, texp*(one-ulp))
				jc = jc - j + 1
				ap.SetRe(jc-1, -(tscal/float64(n+1))/float64(n+2))
				ap.SetRe(jc+j-2-1, one)
				b.SetRe(j-1-1, texp*float64(n*n+n-1))
				texp = texp * two
				jc = jc - j + 2
			}
			b.SetRe(0, (float64(n+1)/float64(n+2))*tscal)
		} else {
			jc = 1
			for j = 1; j <= n-1; j += 2 {
				ap.SetRe(jc+n-j-1, -tscal/float64(n+1))
				ap.SetRe(jc-1, one)
				b.SetRe(j-1, texp*(one-ulp))
				jc = jc + n - j + 1
				ap.SetRe(jc+n-j-1-1, -(tscal/float64(n+1))/float64(n+2))
				ap.SetRe(jc-1, one)
				b.SetRe(j, texp*float64(n*n+n-1))
				texp = texp * two
				jc = jc + n - j
			}
			b.SetRe(n-1, (float64(n+1)/float64(n+2))*tscal)
		}

	} else if imat == 18 {
		//        Type 18:  Generate a unit triangular matrix with elements
		//        between -1 and 1, and make the right hand side large so that it
		//        requires scaling.
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(4, iseed, j-1, ap.Off(jc-1))
				ap.SetRe(jc+j-1-1, zero)
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Zlarnv(4, iseed, n-j, ap.Off(jc))
				}
				ap.SetRe(jc-1, zero)
				jc = jc + n - j + 1
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(2, iseed, n, b)
		iy = goblas.Izamax(n, b.Off(0, 1))
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		goblas.Zdscal(n, bscal, b.Off(0, 1))

	} else if imat == 19 {
		//        Type 19:  Generate a triangular matrix with elements between
		//        BIGNUM/(n-1) and BIGNUM so that at least one of the column
		//        norms will exceed BIGNUM.
		//        1/3/91:  ZLATPS no longer can handle this case
		tleft = bignum / math.Max(one, float64(n-1))
		tscal = bignum * (float64(n-1) / math.Max(one, float64(n)))
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(5, iseed, j, ap.Off(jc-1))
				golapack.Dlarnv(1, iseed, j, rwork)
				for i = 1; i <= j; i++ {
					ap.Set(jc+i-1-1, ap.Get(jc+i-1-1)*complex(tleft+rwork.Get(i-1)*tscal, 0))
				}
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Zlarnv(5, iseed, n-j+1, ap.Off(jc-1))
				golapack.Dlarnv(1, iseed, n-j+1, rwork)
				for i = j; i <= n; i++ {
					ap.Set(jc+i-j-1, ap.Get(jc+i-j-1)*complex(tleft+rwork.Get(i-j)*tscal, 0))
				}
				jc = jc + n - j + 1
			}
		}
		golapack.Zlarnv(2, iseed, n, b)
		goblas.Zdscal(n, two, b.Off(0, 1))
	}

	//     Flip the matrix across its counter-diagonal if the transpose will
	//     be used.
	if trans != NoTrans {
		if upper {
			jj = 1
			jr = n * (n + 1) / 2
			for j = 1; j <= n/2; j++ {
				jl = jj
				for i = j; i <= n-j; i++ {
					t = ap.GetRe(jr - i + j - 1)
					ap.Set(jr-i+j-1, ap.Get(jl-1))
					ap.SetRe(jl-1, t)
					jl = jl + i
				}
				jj = jj + j + 1
				jr = jr - (n - j + 1)
			}
		} else {
			jl = 1
			jj = n * (n + 1) / 2
			for j = 1; j <= n/2; j++ {
				jr = jj
				for i = j; i <= n-j; i++ {
					t = ap.GetRe(jl + i - j - 1)
					ap.Set(jl+i-j-1, ap.Get(jr-1))
					ap.SetRe(jr-1, t)
					jr = jr - i
				}
				jl = jl + n - j + 1
				jj = jj - j - 1
			}
		}
	}

	return
}
