package lin

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/golapack/gltest/matgen"
	"github.com/whipstein/golinalg/mat"
)

// Zlattb generates a triangular test matrix in 2-dimensional storage.
// IMAT and UPLO uniquely specify the properties of the test matrix,
// which is returned in the array A.
func Zlattb(imat *int, uplo, trans byte, diag *byte, iseed *[]int, n, kd *int, ab *mat.CMatrix, ldab *int, b, work *mat.CVector, rwork *mat.Vector, info *int) {
	var upper bool
	var dist, packit, _type byte
	var plus1, plus2, star1 complex128
	var anorm, bignum, bnorm, bscal, cndnum, one, rexp, sfac, smlnum, texp, tleft, tnorm, tscal, two, ulp, unfl, zero float64
	var i, ioff, iy, j, jcount, kl, ku, lenj, mode int

	one = 1.0
	two = 2.0
	zero = 0.0

	path := []byte("ZTB")
	unfl = golapack.Dlamch(SafeMinimum)
	ulp = golapack.Dlamch(Epsilon) * golapack.Dlamch(Base)
	smlnum = unfl
	bignum = (one - ulp) / smlnum
	golapack.Dlabad(&smlnum, &bignum)
	if ((*imat) >= 6 && (*imat) <= 9) || (*imat) == 17 {
		(*diag) = 'U'
	} else {
		(*diag) = 'N'
	}
	(*info) = 0

	//     Quick return if N.LE.0.
	if (*n) <= 0 {
		return
	}

	//     Call ZLATB4 to set parameters for CLATMS.
	upper = uplo == 'U'
	if upper {
		Zlatb4(path, imat, n, n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
		ku = (*kd)
		ioff = 1 + max(0, (*kd)-(*n)+1)
		kl = 0
		packit = 'Q'
	} else {
		Zlatb4(path, toPtr(-(*imat)), n, n, &_type, &kl, &ku, &anorm, &mode, &cndnum, &dist)
		kl = (*kd)
		ioff = 1
		ku = 0
		packit = 'B'
	}

	//     IMAT <= 5:  Non-unit triangular matrix
	if (*imat) <= 5 {
		matgen.Zlatms(n, n, dist, iseed, _type, rwork, &mode, &cndnum, &anorm, &kl, &ku, packit, ab.Off(ioff-1, 0), ldab, work, info)

		//     IMAT > 5:  Unit triangular matrix
		//     The diagonal is deliberately set to something other than 1.
		//
		//     IMAT = 6:  Matrix is the identity
	} else if (*imat) == 6 {
		if upper {
			for j = 1; j <= (*n); j++ {
				for i = max(1, (*kd)+2-j); i <= (*kd); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				ab.SetRe((*kd), j-1, float64(j))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				ab.SetRe(0, j-1, float64(j))
				for i = 2; i <= min((*kd)+1, (*n)-j+1); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
			}
		}

		//     IMAT > 6:  Non-trivial unit triangular matrix
		//
		//     A unit triangular matrix T with condition CNDNUM is formed.
		//     In this version, T only has bandwidth 2, the rest of it is zero.
	} else if (*imat) <= 9 {
		tnorm = math.Sqrt(cndnum)

		//        Initialize AB to zero.
		if upper {
			for j = 1; j <= (*n); j++ {
				for i = max(1, (*kd)+2-j); i <= (*kd); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				ab.SetRe((*kd), j-1, float64(j))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				for i = 2; i <= min((*kd)+1, (*n)-j+1); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				ab.SetRe(0, j-1, float64(j))
			}
		}

		//        Special case:  T is tridiagonal.  Set every other offdiagonal
		//        so that the matrix has norm TNORM+1.
		if (*kd) == 1 {
			if upper {
				ab.Set(0, 1, complex(tnorm, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
				lenj = ((*n) - 3) / 2
				golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, &lenj, work)
				for j = 1; j <= lenj; j++ {
					ab.Set(0, 2*(j+1)-1, complex(tnorm, 0)*work.Get(j-1))
				}
			} else {
				ab.Set(1, 0, complex(tnorm, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
				lenj = ((*n) - 3) / 2
				golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, &lenj, work)
				for j = 1; j <= lenj; j++ {
					ab.Set(1, 2*j, complex(tnorm, 0)*work.Get(j-1))
				}
			}
		} else if (*kd) > 1 {
			//           Form a unit triangular matrix T with condition CNDNUM.  T is
			//           given by
			//                   | 1   +   *                      |
			//                   |     1   +                      |
			//               T = |         1   +   *              |
			//                   |             1   +              |
			//                   |                 1   +   *      |
			//                   |                     1   +      |
			//                   |                          . . . |
			//        Each element marked with a '*' is formed by taking the product
			//        of the adjacent elements marked with '+'.  The '*'s can be
			//        chosen freely, and the '+'s are chosen so that the inverse of
			//        T will have elements of the same magnitude as T.
			//
			//        The two offdiagonals of T are stored in WORK.
			star1 = complex(tnorm, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			sfac = math.Sqrt(tnorm)
			plus1 = complex(sfac, 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
			for j = 1; j <= (*n); j += 2 {
				plus2 = star1 / plus1
				work.Set(j-1, plus1)
				work.Set((*n)+j-1, star1)
				if j+1 <= (*n) {
					work.Set(j, plus2)
					work.SetRe((*n)+j, zero)
					plus1 = star1 / plus2

					//                 Generate a new *-value with norm between math.Sqrt(TNORM)
					//                 and TNORM.
					rexp = matgen.Dlarnd(func() *int { y := 2; return &y }(), iseed)
					if rexp < zero {
						star1 = complex(-math.Pow(sfac, one-rexp), 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
					} else {
						star1 = complex(math.Pow(sfac, one+rexp), 0) * matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)
					}
				}
			}

			//           Copy the tridiagonal T to AB.
			if upper {
				goblas.Zcopy((*n)-1, work.Off(0, 1), ab.CVector((*kd)-1, 1, *ldab))
				goblas.Zcopy((*n)-2, work.Off((*n), 1), ab.CVector((*kd)-1-1, 2, *ldab))
			} else {
				goblas.Zcopy((*n)-1, work.Off(0, 1), ab.CVector(1, 0, *ldab))
				goblas.Zcopy((*n)-2, work.Off((*n), 1), ab.CVector(2, 0, *ldab))
			}
		}

		//     IMAT > 9:  Pathological test cases.  These triangular matrices
		//     are badly scaled or badly conditioned, so when used in solving a
		//     triangular system they may cause overflow in the solution vector.
	} else if (*imat) == 10 {
		//        Type 10:  Generate a triangular matrix with elements between
		//        -1 and 1. Give the diagonal norm 2 to make it well-conditioned.
		//        Make the right hand side large so that it requires scaling.
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j-1, *kd)
				golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector((*kd)+1-lenj-1, j-1))
				ab.Set((*kd), j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)*complex(two, 0))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector(1, j-1))
				}
				ab.Set(0, j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)*complex(two, 0))
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		iy = goblas.Izamax(*n, b.Off(0, 1))
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		goblas.Zdscal(*n, bscal, b.Off(0, 1))

	} else if (*imat) == 11 {
		//        Type 11:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 11, the offdiagonal elements are small (CNORM(j) < 1).
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		tscal = one / float64((*kd)+1)
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j-1, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector((*kd)+2-lenj-1, j-1))
					goblas.Zdscal(lenj, tscal, ab.CVector((*kd)+2-lenj-1, j-1, 1))
				}
				ab.Set((*kd), j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			ab.Set((*kd), (*n)-1, complex(smlnum, 0)*ab.Get((*kd), (*n)-1))
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector(1, j-1))
					goblas.Zdscal(lenj, tscal, ab.CVector(1, j-1, 1))
				}
				ab.Set(0, j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			ab.Set(0, 0, complex(smlnum, 0)*ab.Get(0, 0))
		}

	} else if (*imat) == 12 {
		//        Type 12:  Make the first diagonal element in the solve small to
		//        cause immediate overflow when dividing by T(j,j).
		//        In _type 12, the offdiagonal elements are O(1) (CNORM(j) > 1).
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j-1, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector((*kd)+2-lenj-1, j-1))
				}
				ab.Set((*kd), j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			ab.Set((*kd), (*n)-1, complex(smlnum, 0)*ab.Get((*kd), (*n)-1))
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector(1, j-1))
				}
				ab.Set(0, j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			ab.Set(0, 0, complex(smlnum, 0)*ab.Get(0, 0))
		}

	} else if (*imat) == 13 {
		//        Type 13:  T is diagonal with small numbers on the diagonal to
		//        make the growth factor underflow, but a small right hand side
		//        chosen so that the solution does not overflow.
		if upper {
			jcount = 1
			for j = (*n); j >= 1; j -= 1 {
				for i = max(1, (*kd)+1-(j-1)); i <= (*kd); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				if jcount <= 2 {
					ab.Set((*kd), j-1, complex(smlnum, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
				} else {
					ab.Set((*kd), j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
				}
				jcount = jcount + 1
				if jcount > 4 {
					jcount = 1
				}
			}
		} else {
			jcount = 1
			for j = 1; j <= (*n); j++ {
				for i = 2; i <= min((*n)-j+1, (*kd)+1); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				if jcount <= 2 {
					ab.Set(0, j-1, complex(smlnum, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
				} else {
					ab.Set(0, j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
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
			for i = (*n); i >= 2; i -= 2 {
				b.SetRe(i-1, zero)
				b.Set(i-1-1, complex(smlnum, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
		} else {
			b.SetRe((*n)-1, zero)
			for i = 1; i <= (*n)-1; i += 2 {
				b.SetRe(i-1, zero)
				b.Set(i, complex(smlnum, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
		}

	} else if (*imat) == 14 {
		//        Type 14:  Make the diagonal elements small to cause gradual
		//        overflow when dividing by T(j,j).  To control the amount of
		//        scaling needed, the matrix is bidiagonal.
		texp = one / float64((*kd)+1)
		tscal = math.Pow(smlnum, texp)
		golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, n, b)
		if upper {
			for j = 1; j <= (*n); j++ {
				for i = max(1, (*kd)+2-j); i <= (*kd); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				if j > 1 && (*kd) > 0 {
					ab.Set((*kd)-1, j-1, complex(-one, -one))
				}
				ab.Set((*kd), j-1, complex(tscal, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			b.Set((*n)-1, complex(one, one))
		} else {
			for j = 1; j <= (*n); j++ {
				for i = 3; i <= min((*n)-j+1, (*kd)+1); i++ {
					ab.SetRe(i-1, j-1, zero)
				}
				if j < (*n) && (*kd) > 0 {
					ab.Set(1, j-1, complex(-one, -one))
				}
				ab.Set(0, j-1, complex(tscal, 0)*matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed))
			}
			b.Set(0, complex(one, one))
		}

	} else if (*imat) == 15 {
		//        Type 15:  One zero diagonal element.
		iy = (*n)/2 + 1
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j, (*kd)+1)
				golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector((*kd)+2-lenj-1, j-1))
				if j != iy {
					ab.Set((*kd), j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)*complex(two, 0))
				} else {
					ab.SetRe((*kd), j-1, zero)
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j+1, (*kd)+1)
				golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector(0, j-1))
				if j != iy {
					ab.Set(0, j-1, matgen.Zlarnd(func() *int { y := 5; return &y }(), iseed)*complex(two, 0))
				} else {
					ab.SetRe(0, j-1, zero)
				}
			}
		}
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		goblas.Zdscal(*n, two, b.Off(0, 1))

	} else if (*imat) == 16 {
		//        Type 16:  Make the offdiagonal elements large to cause overflow
		//        when adding a column of T.  In the non-transposed case, the
		//        matrix is constructed to cause overflow when adding a column in
		//        every other step.
		tscal = unfl / ulp
		tscal = (one - ulp) / tscal
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= (*kd)+1; i++ {
				ab.SetRe(i-1, j-1, zero)
			}
		}
		texp = one
		if (*kd) > 0 {
			if upper {
				for j = (*n); j >= 1; j -= (*kd) {
					for i = j; i >= max(1, j-(*kd)+1); i -= 2 {
						ab.SetRe(1+(j-i)-1, i-1, -tscal/float64((*kd)+2))
						ab.SetRe((*kd), i-1, one)
						b.SetRe(i-1, texp*(one-ulp))
						if i > max(1, j-(*kd)+1) {
							ab.SetRe(2+(j-i)-1, i-1-1, -(tscal/float64((*kd)+2))/float64((*kd)+3))
							ab.SetRe((*kd), i-1-1, one)
							b.SetRe(i-1-1, texp*float64(((*kd)+1)*((*kd)+1)+(*kd)))
						}
						texp = texp * two
					}
					b.SetRe(max(1, j-(*kd)+1)-1, (float64((*kd)+2)/float64((*kd)+3))*tscal)
				}
			} else {
				for j = 1; j <= (*n); j += (*kd) {
					texp = one
					lenj = min((*kd)+1, (*n)-j+1)
					for i = j; i <= min(*n, j+(*kd)-1); i += 2 {
						ab.SetRe(lenj-(i-j)-1, j-1, -tscal/float64((*kd)+2))
						ab.SetRe(0, j-1, one)
						b.SetRe(j-1, texp*(one-ulp))
						if i < min(*n, j+(*kd)-1) {
							ab.SetRe(lenj-(i-j+1)-1, i, -(tscal/float64((*kd)+2))/float64((*kd)+3))
							ab.SetRe(0, i, one)
							b.SetRe(i, texp*float64(((*kd)+1)*((*kd)+1)+(*kd)))
						}
						texp = texp * two
					}
					b.SetRe(min(*n, j+(*kd)-1)-1, (float64((*kd)+2)/float64((*kd)+3))*tscal)
				}
			}
		}

	} else if (*imat) == 17 {
		//        Type 17:  Generate a unit triangular matrix with elements
		//        between -1 and 1, and make the right hand side large so that it
		//        requires scaling.
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j-1, *kd)
				golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector((*kd)+1-lenj-1, j-1))
				ab.SetRe((*kd), j-1, float64(j))
			}
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j, *kd)
				if lenj > 0 {
					golapack.Zlarnv(func() *int { y := 4; return &y }(), iseed, &lenj, ab.CVector(1, j-1))
				}
				ab.SetRe(0, j-1, float64(j))
			}
		}

		//        Set the right hand side so that the largest value is BIGNUM.
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		iy = goblas.Izamax(*n, b.Off(0, 1))
		bnorm = b.GetMag(iy - 1)
		bscal = bignum / math.Max(one, bnorm)
		goblas.Zdscal(*n, bscal, b.Off(0, 1))

	} else if (*imat) == 18 {
		//        Type 18:  Generate a triangular matrix with elements between
		//        BIGNUM/(KD+1) and BIGNUM so that at least one of the column
		//        norms will exceed BIGNUM.
		//        1/3/91:  ZLATBS no longer can handle this case
		tleft = bignum / float64((*kd)+1)
		tscal = bignum * (float64((*kd)+1) / float64((*kd)+2))
		if upper {
			for j = 1; j <= (*n); j++ {
				lenj = min(j, (*kd)+1)
				golapack.Zlarnv(func() *int { y := 5; return &y }(), iseed, &lenj, ab.CVector((*kd)+2-lenj-1, j-1))
				golapack.Dlarnv(func() *int { y := 1; return &y }(), iseed, &lenj, rwork.Off((*kd)+2-lenj-1))
				for i = (*kd) + 2 - lenj; i <= (*kd)+1; i++ {
					ab.Set(i-1, j-1, ab.Get(i-1, j-1)*complex(tleft+rwork.Get(i-1)*tscal, 0))
				}
			}
		} else {
			for j = 1; j <= (*n); j++ {
				lenj = min((*n)-j+1, (*kd)+1)
				golapack.Zlarnv(func() *int { y := 5; return &y }(), iseed, &lenj, ab.CVector(0, j-1))
				golapack.Dlarnv(func() *int { y := 1; return &y }(), iseed, &lenj, rwork)
				for i = 1; i <= lenj; i++ {
					ab.Set(i-1, j-1, ab.Get(i-1, j-1)*complex(tleft+rwork.Get(i-1)*tscal, 0))
				}
			}
		}
		golapack.Zlarnv(func() *int { y := 2; return &y }(), iseed, n, b)
		goblas.Zdscal(*n, two, b.Off(0, 1))
	}

	//     Flip the matrix if the transpose will be used.
	if trans != 'N' {
		if upper {
			for j = 1; j <= (*n)/2; j++ {
				lenj = min((*n)-2*j+1, (*kd)+1)
				goblas.Zswap(lenj, ab.CVector((*kd), j-1, (*ldab)-1), ab.CVector((*kd)+2-lenj-1, (*n)-j, -1))
			}
		} else {
			for j = 1; j <= (*n)/2; j++ {
				lenj = min((*n)-2*j+1, (*kd)+1)
				goblas.Zswap(lenj, ab.CVector(0, j-1, 1), ab.CVector(lenj-1, (*n)-j+2-lenj-1, -(*ldab)+1))
			}
		}
	}
}
