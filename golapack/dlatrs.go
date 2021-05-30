package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Dlatrs solves one of the triangular systems
//
//    A *x = s*b  or  A**T *x = s*b
//
// with scaling to prevent overflow.  Here A is an upper or lower
// triangular matrix, A**T denotes the transpose of A, x and b are
// n-element vectors, and s is a scaling factor, usually less than
// or equal to 1, chosen so that the components of x will be less than
// the overflow threshold.  If the unscaled problem will not cause
// overflow, the Level 2 BLAS routine DTRSV is called.  If the matrix A
// is singular (A(j,j) = 0 for some j), then s is set to 0 and a
// non-trivial solution to A*x = 0 is returned.
func Dlatrs(uplo, trans, diag, normin byte, n *int, a *mat.Matrix, lda *int, x *mat.Vector, scale *float64, cnorm *mat.Vector, info *int) {
	var notran, nounit, upper bool
	var bignum, grow, half, one, rec, smlnum, sumj, tjj, tjjs, tmax, tscal, uscal, xbnd, xj, xmax, zero float64
	var i, imax, j, jfirst, jinc, jlast int

	zero = 0.0
	half = 0.5
	one = 1.0

	(*info) = 0
	upper = uplo == 'U'
	notran = trans == 'N'
	nounit = diag == 'N'

	//     Test the input parameters.
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if !notran && trans != 'T' && trans != 'C' {
		(*info) = -2
	} else if !nounit && diag != 'U' {
		(*info) = -3
	} else if normin != 'Y' && normin != 'N' {
		(*info) = -4
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*lda) < maxint(1, *n) {
		(*info) = -7
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLATRS"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Determine machine dependent parameters to control overflow.
	smlnum = Dlamch(SafeMinimum) / Dlamch(Precision)
	bignum = one / smlnum
	(*scale) = one

	if normin == 'N' {
		//        Compute the 1-norm of each column, not including the diagonal.
		if upper {
			//           A is upper triangular.
			for j = 1; j <= (*n); j++ {
				cnorm.Set(j-1, goblas.Dasum(toPtr(j-1), a.Vector(0, j-1), func() *int { y := 1; return &y }()))
			}
		} else {
			//           A is lower triangular.
			for j = 1; j <= (*n)-1; j++ {
				cnorm.Set(j-1, goblas.Dasum(toPtr((*n)-j), a.Vector(j+1-1, j-1), func() *int { y := 1; return &y }()))
			}
			cnorm.Set((*n)-1, zero)
		}
	}

	//     Scale the column norms by TSCAL if the maximum element in CNORM is
	//     greater than BIGNUM.
	imax = goblas.Idamax(n, cnorm, func() *int { y := 1; return &y }())
	tmax = cnorm.Get(imax - 1)
	if tmax <= bignum {
		tscal = one
	} else {
		tscal = one / (smlnum * tmax)
		goblas.Dscal(n, &tscal, cnorm, func() *int { y := 1; return &y }())
	}

	//     Compute a bound on the computed solution vector to see if the
	//     Level 2 BLAS routine DTRSV can be used.
	j = goblas.Idamax(n, x, func() *int { y := 1; return &y }())
	xmax = math.Abs(x.Get(j - 1))
	xbnd = xmax
	if notran {
		//        Compute the growth in A * x = b.
		if upper {
			jfirst = (*n)
			jlast = 1
			jinc = -1
		} else {
			jfirst = 1
			jlast = (*n)
			jinc = 1
		}

		if tscal != one {
			grow = zero
			goto label50
		}

		if nounit {
			//           A is non-unit triangular.
			//
			//           Compute GROW = 1/G(j) and XBND = 1/M(j).
			//           Initially, G(0) = max{x(i), i=1,...,n}.
			grow = one / maxf64(xbnd, smlnum)
			xbnd = grow
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label50
				}

				//              M(j) = G(j-1) / math.Abs(A(j,j))
				tjj = math.Abs(a.Get(j-1, j-1))
				xbnd = minf64(xbnd, minf64(one, tjj)*grow)
				if tjj+cnorm.Get(j-1) >= smlnum {
					//                 G(j) = G(j-1)*( 1 + CNORM(j) / math.Abs(A(j,j)) )
					grow = grow * (tjj / (tjj + cnorm.Get(j-1)))
				} else {
					//                 G(j) could overflow, set GROW to 0.
					grow = zero
				}
			}
			grow = xbnd
		} else {
			//           A is unit triangular.
			//
			//           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
			grow = minf64(one, one/maxf64(xbnd, smlnum))
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label50
				}

				//              G(j) = G(j-1)*( 1 + CNORM(j) )
				grow = grow * (one / (one + cnorm.Get(j-1)))
			}
		}
	label50:
	} else {
		//        Compute the growth in A**T * x = b.
		if upper {
			jfirst = 1
			jlast = (*n)
			jinc = 1
		} else {
			jfirst = (*n)
			jlast = 1
			jinc = -1
		}

		if tscal != one {
			grow = zero
			goto label80
		}

		if nounit {
			//           A is non-unit triangular.
			//
			//           Compute GROW = 1/G(j) and XBND = 1/M(j).
			//           Initially, M(0) = max{x(i), i=1,...,n}.
			grow = one / maxf64(xbnd, smlnum)
			xbnd = grow
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label80
				}

				//              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
				xj = one + cnorm.Get(j-1)
				grow = minf64(grow, xbnd/xj)

				//              M(j) = M(j-1)*( 1 + CNORM(j) ) / math.Abs(A(j,j))
				tjj = math.Abs(a.Get(j-1, j-1))
				if xj > tjj {
					xbnd = xbnd * (tjj / xj)
				}
			}
			grow = minf64(grow, xbnd)
		} else {
			//           A is unit triangular.
			//
			//           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
			grow = minf64(one, one/maxf64(xbnd, smlnum))
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label80
				}

				//              G(j) = ( 1 + CNORM(j) )*G(j-1)
				xj = one + cnorm.Get(j-1)
				grow = grow / xj
			}
		}
	label80:
	}

	if (grow * tscal) > smlnum {
		//        Use the Level 2 BLAS solve if the reciprocal of the bound on
		//        elements of X is not too small.
		goblas.Dtrsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), n, a, lda, x, func() *int { y := 1; return &y }())
	} else {
		//        Use a Level 1 BLAS solve, scaling intermediate results.
		if xmax > bignum {
			//           Scale X so that its components are less than or equal to
			//           BIGNUM in absolute value.
			(*scale) = bignum / xmax
			goblas.Dscal(n, scale, x, func() *int { y := 1; return &y }())
			xmax = bignum
		}

		if notran {
			//           Solve A * x = b
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Compute x(j) = b(j) / A(j,j), scaling x if necessary.
				xj = math.Abs(x.Get(j - 1))
				if nounit {
					tjjs = a.Get(j-1, j-1) * tscal
				} else {
					tjjs = tscal
					if tscal == one {
						goto label100
					}
				}
				tjj = math.Abs(tjjs)
				if tjj > smlnum {
					//                    math.Abs(A(j,j)) > SMLNUM:
					if tjj < one {
						if xj > tjj*bignum {
							//                          Scale x by 1/b(j).
							rec = one / xj
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					x.Set(j-1, x.Get(j-1)/tjjs)
					xj = math.Abs(x.Get(j - 1))
				} else if tjj > zero {
					//                    0 < math.Abs(A(j,j)) <= SMLNUM:
					if xj > tjj*bignum {
						//                       Scale x by (1/math.Abs(x(j)))*math.Abs(A(j,j))*BIGNUM
						//                       to avoid overflow when dividing by A(j,j).
						rec = (tjj * bignum) / xj
						if cnorm.Get(j-1) > one {
							//                          Scale by 1/CNORM(j) to avoid overflow when
							//                          multiplying x(j) times column j.
							rec = rec / cnorm.Get(j-1)
						}
						goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
						(*scale) = (*scale) * rec
						xmax = xmax * rec
					}
					x.Set(j-1, x.Get(j-1)/tjjs)
					xj = math.Abs(x.Get(j - 1))
				} else {
					//                    A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
					//                    scale = 0, and compute a solution to A*x = 0.
					for i = 1; i <= (*n); i++ {
						x.Set(i-1, zero)
					}
					x.Set(j-1, one)
					xj = one
					(*scale) = zero
					xmax = zero
				}
			label100:
				;

				//              Scale x if necessary to avoid overflow when adding a
				//              multiple of column j of A.
				if xj > one {
					rec = one / xj
					if cnorm.Get(j-1) > (bignum-xmax)*rec {
						//                    Scale x by 1/(2*math.Abs(x(j))).
						rec = rec * half
						goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
						(*scale) = (*scale) * rec
					}
				} else if xj*cnorm.Get(j-1) > (bignum - xmax) {
					//                 Scale x by 1/2.
					goblas.Dscal(n, &half, x, func() *int { y := 1; return &y }())
					(*scale) = (*scale) * half
				}

				if upper {
					if j > 1 {
						//                    Compute the update
						//                       x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
						goblas.Daxpy(toPtr(j-1), toPtrf64(-x.Get(j-1)*tscal), a.Vector(0, j-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
						i = goblas.Idamax(toPtr(j-1), x, func() *int { y := 1; return &y }())
						xmax = math.Abs(x.Get(i - 1))
					}
				} else {
					if j < (*n) {
						//                    Compute the update
						//                       x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
						goblas.Daxpy(toPtr((*n)-j), toPtrf64(-x.Get(j-1)*tscal), a.Vector(j+1-1, j-1), func() *int { y := 1; return &y }(), x.Off(j+1-1), func() *int { y := 1; return &y }())
						i = j + goblas.Idamax(toPtr((*n)-j), x.Off(j+1-1), func() *int { y := 1; return &y }())
						xmax = math.Abs(x.Get(i - 1))
					}
				}
			}

		} else {
			//           Solve A**T * x = b
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Compute x(j) = b(j) - sum A(k,j)*x(k).
				//                                    k<>j
				xj = math.Abs(x.Get(j - 1))
				uscal = tscal
				rec = one / maxf64(xmax, one)
				if cnorm.Get(j-1) > (bignum-xj)*rec {
					//                 If x(j) could overflow, scale x by 1/(2*XMAX).
					rec = rec * half
					if nounit {
						tjjs = a.Get(j-1, j-1) * tscal
					} else {
						tjjs = tscal
					}
					tjj = math.Abs(tjjs)
					if tjj > one {
						//                       Divide by A(j,j) when scaling x if A(j,j) > 1.
						rec = minf64(one, rec*tjj)
						uscal = uscal / tjjs
					}
					if rec < one {
						goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
						(*scale) = (*scale) * rec
						xmax = xmax * rec
					}
				}

				sumj = zero
				if uscal == one {
					//                 If the scaling needed for A in the dot product is 1,
					//                 call DDOT to perform the dot product.
					if upper {
						sumj = goblas.Ddot(toPtr(j-1), a.Vector(0, j-1), func() *int { y := 1; return &y }(), x, func() *int { y := 1; return &y }())
					} else if j < (*n) {
						sumj = goblas.Ddot(toPtr((*n)-j), a.Vector(j+1-1, j-1), func() *int { y := 1; return &y }(), x.Off(j+1-1), func() *int { y := 1; return &y }())
					}
				} else {
					//                 Otherwise, use in-line code for the dot product.
					if upper {
						for i = 1; i <= j-1; i++ {
							sumj = sumj + (a.Get(i-1, j-1)*uscal)*x.Get(i-1)
						}
					} else if j < (*n) {
						for i = j + 1; i <= (*n); i++ {
							sumj = sumj + (a.Get(i-1, j-1)*uscal)*x.Get(i-1)
						}
					}
				}

				if uscal == tscal {
					//                 Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j)
					//                 was not used to scale the dotproduct.
					x.Set(j-1, x.Get(j-1)-sumj)
					xj = math.Abs(x.Get(j - 1))
					if nounit {
						tjjs = a.Get(j-1, j-1) * tscal
					} else {
						tjjs = tscal
						if tscal == one {
							goto label150
						}
					}

					//                    Compute x(j) = x(j) / A(j,j), scaling if necessary.
					tjj = math.Abs(tjjs)
					if tjj > smlnum {
						//                       math.Abs(A(j,j)) > SMLNUM:
						if tjj < one {
							if xj > tjj*bignum {
								//                             Scale X by 1/math.Abs(x(j)).
								rec = one / xj
								goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
								(*scale) = (*scale) * rec
								xmax = xmax * rec
							}
						}
						x.Set(j-1, x.Get(j-1)/tjjs)
					} else if tjj > zero {
						//                       0 < math.Abs(A(j,j)) <= SMLNUM:
						if xj > tjj*bignum {
							//                          Scale x by (1/math.Abs(x(j)))*math.Abs(A(j,j))*BIGNUM.
							rec = (tjj * bignum) / xj
							goblas.Dscal(n, &rec, x, func() *int { y := 1; return &y }())
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
						x.Set(j-1, x.Get(j-1)/tjjs)
					} else {
						//                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
						//                       scale = 0, and compute a solution to A**T*x = 0.
						for i = 1; i <= (*n); i++ {
							x.Set(i-1, zero)
						}
						x.Set(j-1, one)
						(*scale) = zero
						xmax = zero
					}
				label150:
				} else {
					//                 Compute x(j) := x(j) / A(j,j)  - sumj if the dot
					//                 product has already been divided by 1/A(j,j).
					x.Set(j-1, x.Get(j-1)/tjjs-sumj)
				}
				xmax = maxf64(xmax, math.Abs(x.Get(j-1)))
			}
		}
		(*scale) = (*scale) / tscal
	}

	//     Scale the column norms by 1/TSCAL for return.
	if tscal != one {
		goblas.Dscal(n, toPtrf64(one/tscal), cnorm, func() *int { y := 1; return &y }())
	}
}
