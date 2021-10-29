package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// dlattp generates a triangular test matrix in packed storage.
// IMAT and UPLO uniquely specify the properties of the test
// matrix, which is returned in the array AP.
func dlattp(imat int, uplo mat.MatUplo, trans mat.MatTrans, iseed []int, n int, a, b, work *mat.Vector) (mat.MatDiag, []int, error) {
	var upper bool
	var dist, packit, _type byte
	var diag mat.MatDiag
	var anorm, bignum, bnorm, bscal, c, cndnum, one, plus1, plus2, ra, rb, rexp, s, sfac, smlnum, star1, stemp, t, texp, tleft, tscal, two, ulp, unfl, x, y, z, zero float64
	var i, iy, j, jc, jcnext, jcount, jj, jl, jr, jx, kl, ku, mode int
	var err error

	one = 1.0
	two = 2.0
	zero = 0.0

	path := "Dtp"
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
		return diag, iseed, err
	}

	//     Call DLATB4 to set parameters for SLATMS.
	upper = uplo == Upper
	if upper {
		_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, imat, n, n)
		packit = 'C'
	} else {
		_type, kl, ku, anorm, mode, cndnum, dist = dlatb4(path, -imat, n, n)
		packit = 'R'
	}

	//     IMAT <= 6:  Non-unit triangular matrix
	if imat <= 6 {
		_, _ = matgen.Dlatms(n, n, dist, &iseed, _type, b, mode, cndnum, anorm, kl, ku, packit, a.Matrix(n, opts), work)

		//     IMAT > 6:  Unit triangular matrix
		//     The diagonal is deliberately set to something other than 1.
		//
		//     IMAT = 7:  Matrix is the identity
	} else if imat == 7 {
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-1; i++ {
					a.Set(jc+i-1-1, zero)
				}
				a.Set(jc+j-1-1, float64(j))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				a.Set(jc-1, float64(j))
				for i = j + 1; i <= n; i++ {
					a.Set(jc+i-j-1, zero)
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
					a.Set(jc+i-1, zero)
				}
				a.Set(jc+j-1, float64(j))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				a.Set(jc-1, float64(j))
				for i = j + 1; i <= n; i++ {
					a.Set(jc+i-j-1, zero)
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
		star1 = 0.25
		sfac = 0.5
		plus1 = sfac
		for j = 1; j <= n; j += 2 {
			plus2 = star1 / plus1
			work.Set(j-1, plus1)
			work.Set(n+j-1, star1)
			if j+1 <= n {
				work.Set(j, plus2)
				work.Set(n+j, zero)
				plus1 = star1 / plus2
				rexp = matgen.Dlarnd(2, &iseed)
				star1 = star1 * math.Pow(sfac, rexp)
				if rexp < zero {
					star1 = -math.Pow(sfac, one-rexp)
				} else {
					star1 = math.Pow(sfac, one+rexp)
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
				a.Set(jc, y)
				if j > 2 {
					a.Set(jc+j-1-1, work.Get(j-2-1))
				}
				if j > 3 {
					a.Set(jc+j-2-1, work.Get(n+j-3-1))
				}
				jc = jc + j
			}
			jc = jc - n
			a.Set(jc, z)
			for j = 2; j <= n-1; j++ {
				a.Set(jc+j-1, y)
			}
		} else {
			//           Set the lower triangle of A with a unit triangular matrix
			//           of known condition number.
			for i = 2; i <= n-1; i++ {
				a.Set(i-1, y)
			}
			a.Set(n-1, z)
			jc = n + 1
			for j = 2; j <= n-1; j++ {
				a.Set(jc, work.Get(j-1-1))
				if j < n-1 {
					a.Set(jc+2-1, work.Get(n+j-1-1))
				}
				a.Set(jc+n-j-1, y)
				jc = jc + n - j + 1
			}
		}

		//        Fill in the zeros using Givens rotations
		if upper {
			jc = 1
			for j = 1; j <= n-1; j++ {
				jcnext = jc + j
				ra = a.Get(jcnext + j - 1 - 1)
				rb = two
				ra, rb, c, s = goblas.Drotg(ra, rb, c, s)

				//              Multiply by [ c  s; -s  c] on the left.
				if n > j+1 {
					jx = jcnext + j
					for i = j + 2; i <= n; i++ {
						stemp = c*a.Get(jx+j-1) + s*a.Get(jx+j)
						a.Set(jx+j, -s*a.Get(jx+j-1)+c*a.Get(jx+j))
						a.Set(jx+j-1, stemp)
						jx = jx + i
					}
				}

				//              Multiply by [-c -s;  s -c] on the right.
				if j > 1 {
					goblas.Drot(j-1, a.Off(jcnext-1, 1), a.Off(jc-1, 1), -c, -s)
				}

				//              Negate A(J,J+1).
				a.Set(jcnext+j-1-1, -a.Get(jcnext+j-1-1))
				jc = jcnext
			}
		} else {
			jc = 1
			for j = 1; j <= n-1; j++ {
				jcnext = jc + n - j + 1
				ra = a.Get(jc + 1 - 1)
				rb = two
				ra, rb, c, s = goblas.Drotg(ra, rb, c, s)

				//              Multiply by [ c -s;  s  c] on the right.
				if n > j+1 {
					goblas.Drot(n-j-1, a.Off(jcnext, 1), a.Off(jc+2-1, 1), c, -s)
				}

				//              Multiply by [-c  s; -s -c] on the left.
				if j > 1 {
					jx = 1
					for i = 1; i <= j-1; i++ {
						stemp = -c*a.Get(jx+j-i-1) + s*a.Get(jx+j-i)
						a.Set(jx+j-i, -s*a.Get(jx+j-i-1)-c*a.Get(jx+j-i))
						a.Set(jx+j-i-1, stemp)
						jx = jx + n - i + 1
					}
				}

				//              Negate A(J+1,J).
				a.Set(jc, -a.Get(jc))
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
				golapack.Dlarnv(2, &iseed, j, a.Off(jc-1))
				a.Set(jc+j-1-1, math.Copysign(two, a.Get(jc+j-1-1)))
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, n-j+1, a.Off(jc-1))
				a.Set(jc-1, math.Copysign(two, a.Get(jc-1)))
				jc = jc + n - j + 1
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Dlarnv(2, &iseed, n, b)
		iy = goblas.Idamax(n, b.Off(0, 1))
		bnorm = math.Abs(b.Get(iy - 1))
		bscal = bignum / math.Max(one, bnorm)
		goblas.Dscal(n, bscal, b.Off(0, 1))

	} else if imat == 12 {
		//        Type 12:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 12, the offdiagonal elements are small (CNORM(j) < 1).
		golapack.Dlarnv(2, &iseed, n, b)
		tscal = one / math.Max(one, float64(n-1))
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, j-1, a.Off(jc-1))
				goblas.Dscal(j-1, tscal, a.Off(jc-1, 1))
				a.Set(jc+j-1-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
				jc = jc + j
			}
			a.Set(n*(n+1)/2-1, smlnum)
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, n-j, a.Off(jc))
				goblas.Dscal(n-j, tscal, a.Off(jc, 1))
				a.Set(jc-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
				jc = jc + n - j + 1
			}
			a.Set(0, smlnum)
		}

	} else if imat == 13 {
		//        Type 13:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 13, the offdiagonal elements are O(1) (CNORM(j) > 1).
		golapack.Dlarnv(2, &iseed, n, b)
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, j-1, a.Off(jc-1))
				a.Set(jc+j-1-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
				jc = jc + j
			}
			a.Set(n*(n+1)/2-1, smlnum)
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, n-j, a.Off(jc))
				a.Set(jc-1, math.Copysign(one, matgen.Dlarnd(2, &iseed)))
				jc = jc + n - j + 1
			}
			a.Set(0, smlnum)
		}

	} else if imat == 14 {
		//        Type 14:  T is diagonal with small numbers on the diagonal to
		//        make the growth factor underflow, but a small right hand side
		//        chosen so that the solution does not overflow.
		if upper {
			jcount = 1
			jc = (n-1)*n/2 + 1
			for j = n; j >= 1; j-- {
				for i = 1; i <= j-1; i++ {
					a.Set(jc+i-1-1, zero)
				}
				if jcount <= 2 {
					a.Set(jc+j-1-1, smlnum)
				} else {
					a.Set(jc+j-1-1, one)
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
					a.Set(jc+i-j-1, zero)
				}
				if jcount <= 2 {
					a.Set(jc-1, smlnum)
				} else {
					a.Set(jc-1, one)
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
			b.Set(0, zero)
			for i = n; i >= 2; i -= 2 {
				b.Set(i-1, zero)
				b.Set(i-1-1, smlnum)
			}
		} else {
			b.Set(n-1, zero)
			for i = 1; i <= n-1; i += 2 {
				b.Set(i-1, zero)
				b.Set(i, smlnum)
			}
		}

	} else if imat == 15 {
		//        Type 15:  Make the diagonal elements small to cause gradual
		//        overflow when dividing by T(j,j).  To control the amount of
		//        scaling needed, the matrix is bidiagonal.
		texp = one / math.Max(one, float64(n-1))
		tscal = math.Pow(smlnum, texp)
		golapack.Dlarnv(2, &iseed, n, b)
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				for i = 1; i <= j-2; i++ {
					a.Set(jc+i-1-1, zero)
				}
				if j > 1 {
					a.Set(jc+j-2-1, -one)
				}
				a.Set(jc+j-1-1, tscal)
				jc = jc + j
			}
			b.Set(n-1, one)
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				for i = j + 2; i <= n; i++ {
					a.Set(jc+i-j-1, zero)
				}
				if j < n {
					a.Set(jc, -one)
				}
				a.Set(jc-1, tscal)
				jc = jc + n - j + 1
			}
			b.Set(0, one)
		}

	} else if imat == 16 {
		//        Type 16:  One zero diagonal element.
		iy = n/2 + 1
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, j, a.Off(jc-1))
				if j != iy {
					a.Set(jc+j-1-1, math.Copysign(two, a.Get(jc+j-1-1)))
				} else {
					a.Set(jc+j-1-1, zero)
				}
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, n-j+1, a.Off(jc-1))
				if j != iy {
					a.Set(jc-1, math.Copysign(two, a.Get(jc-1)))
				} else {
					a.Set(jc-1, zero)
				}
				jc = jc + n - j + 1
			}
		}
		golapack.Dlarnv(2, &iseed, n, b)
		goblas.Dscal(n, two, b.Off(0, 1))

	} else if imat == 17 {
		//        Type 17:  Make the offdiagonal elements large to cause overflow
		//        when adding a column of T.  In the non-transposed case, the
		//        matrix is constructed to cause overflow when adding a column in
		//        every other step.
		tscal = unfl / ulp
		tscal = (one - ulp) / tscal
		for j = 1; j <= n*(n+1)/2; j++ {
			a.Set(j-1, zero)
		}
		texp = one
		if upper {
			jc = (n-1)*n/2 + 1
			for j = n; j >= 2; j -= 2 {
				a.Set(jc-1, -tscal/float64(n+1))
				a.Set(jc+j-1-1, one)
				b.Set(j-1, texp*(one-ulp))
				jc = jc - j + 1
				a.Set(jc-1, -(tscal/float64(n+1))/float64(n+2))
				a.Set(jc+j-2-1, one)
				b.Set(j-1-1, texp*float64(n*n+n-1))
				texp = texp * two
				jc = jc - j + 2
			}
			b.Set(0, (float64(n+1)/float64(n+2))*tscal)
		} else {
			jc = 1
			for j = 1; j <= n-1; j += 2 {
				a.Set(jc+n-j-1, -tscal/float64(n+1))
				a.Set(jc-1, one)
				b.Set(j-1, texp*(one-ulp))
				jc = jc + n - j + 1
				a.Set(jc+n-j-1-1, -(tscal/float64(n+1))/float64(n+2))
				a.Set(jc-1, one)
				b.Set(j, texp*float64(n*n+n-1))
				texp = texp * two
				jc = jc + n - j
			}
			b.Set(n-1, (float64(n+1)/float64(n+2))*tscal)
		}

	} else if imat == 18 {
		//        Type 18:  Generate a unit triangular matrix with elements
		//        between -1 and 1, and make the right hand side large so that it
		//        requires scaling.
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, j-1, a.Off(jc-1))
				a.Set(jc+j-1-1, zero)
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				if j < n {
					golapack.Dlarnv(2, &iseed, n-j, a.Off(jc))
				}
				a.Set(jc-1, zero)
				jc = jc + n - j + 1
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Dlarnv(2, &iseed, n, b)
		iy = goblas.Idamax(n, b.Off(0, 1))
		bnorm = math.Abs(b.Get(iy - 1))
		bscal = bignum / math.Max(one, bnorm)
		goblas.Dscal(n, bscal, b.Off(0, 1))

	} else if imat == 19 {
		//        Type 19:  Generate a triangular matrix with elements between
		//        BIGNUM/(n-1) and BIGNUM so that at least one of the column
		//        norms will exceed BIGNUM.
		tleft = bignum / math.Max(one, float64(n-1))
		tscal = bignum * (float64(n-1) / math.Max(one, float64(n)))
		if upper {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, j, a.Off(jc-1))
				for i = 1; i <= j; i++ {
					a.Set(jc+i-1-1, math.Copysign(tleft, a.Get(jc+i-1-1))+tscal*a.Get(jc+i-1-1))
				}
				jc = jc + j
			}
		} else {
			jc = 1
			for j = 1; j <= n; j++ {
				golapack.Dlarnv(2, &iseed, n-j+1, a.Off(jc-1))
				for i = j; i <= n; i++ {
					a.Set(jc+i-j-1, math.Copysign(tleft, a.Get(jc+i-j-1))+tscal*a.Get(jc+i-j-1))
				}
				jc = jc + n - j + 1
			}
		}
		golapack.Dlarnv(2, &iseed, n, b)
		goblas.Dscal(n, two, b.Off(0, 1))
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
					t = a.Get(jr - i + j - 1)
					a.Set(jr-i+j-1, a.Get(jl-1))
					a.Set(jl-1, t)
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
					t = a.Get(jl + i - j - 1)
					a.Set(jl+i-j-1, a.Get(jr-1))
					a.Set(jr-1, t)
					jr = jr - i
				}
				jl = jl + n - j + 1
				jj = jj - j - 1
			}
		}
	}

	return diag, iseed, err
}
