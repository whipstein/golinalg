package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dlatps solves one of the triangular systems
//
//    A *x = s*b  or  A**T*x = s*b
//
// with scaling to prevent overflow, where A is an upper or lower
// triangular matrix stored in packed form.  Here A**T denotes the
// transpose of A, x and b are n-element vectors, and s is a scaling
// factor, usually less than or equal to 1, chosen so that the
// components of x will be less than the overflow threshold.  If the
// unscaled problem will not cause overflow, the Level 2 BLAS routine
// DTPSV is called. If the matrix A is singular (A(j,j) = 0 for some j),
// then s is set to 0 and a non-trivial solution to A*x = 0 is returned.
func Dlatps(uplo, trans, diag, normin byte, n *int, ap, x *mat.Vector, scale *float64, cnorm *mat.Vector, info *int) {
	var notran, nounit, upper bool
	var bignum, grow, half, one, rec, smlnum, sumj, tjj, tjjs, tmax, tscal, uscal, xbnd, xj, xmax, zero float64
	var i, imax, ip, j, jfirst, jinc, jlast, jlen int
	var iter []int
	var err error
	_ = err

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
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DLATPS"), -(*info))
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
			ip = 1
			for j = 1; j <= (*n); j++ {
				cnorm.Set(j-1, goblas.Dasum(j-1, ap.Off(ip-1), 1))
				ip = ip + j
			}
		} else {
			//           A is lower triangular.
			ip = 1
			for j = 1; j <= (*n)-1; j++ {
				cnorm.Set(j-1, goblas.Dasum((*n)-j, ap.Off(ip+1-1), 1))
				ip = ip + (*n) - j + 1
			}
			cnorm.Set((*n)-1, zero)
		}
	}

	//     Scale the column norms by TSCAL if the maximum element in CNORM is
	//     greater than BIGNUM.
	imax = goblas.Idamax(*n, cnorm, 1)
	tmax = cnorm.Get(imax - 1)
	if tmax <= bignum {
		tscal = one
	} else {
		tscal = one / (smlnum * tmax)
		goblas.Dscal(*n, tscal, cnorm, 1)
	}

	//     Compute a bound on the computed solution vector to see if the
	//     Level 2 BLAS routine DTPSV can be used.
	j = goblas.Idamax(*n, x, 1)
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
		iter = genIter(jfirst, jlast, jinc)

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
			ip = jfirst * (jfirst + 1) / 2
			jlen = (*n)
			for _, j = range iter {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label50
				}

				//              M(j) = G(j-1) / abs(A(j,j))
				tjj = math.Abs(ap.Get(ip - 1))
				xbnd = minf64(xbnd, minf64(one, tjj)*grow)
				if tjj+cnorm.Get(j-1) >= smlnum {
					//                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
					grow = grow * (tjj / (tjj + cnorm.Get(j-1)))
				} else {
					//                 G(j) could overflow, set GROW to 0.
					grow = zero
				}
				ip = ip + jinc*jlen
				jlen = jlen - 1
			}
			grow = xbnd
		} else {
			//           A is unit triangular.
			//
			//           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
			grow = minf64(one, one/maxf64(xbnd, smlnum))
			for _, j = range iter {
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
		iter = genIter(jfirst, jlast, jinc)

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
			ip = jfirst * (jfirst + 1) / 2
			jlen = 1
			for _, j = range iter {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label80
				}

				//              G(j) = max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
				xj = one + cnorm.Get(j-1)
				grow = minf64(grow, xbnd/xj)

				//              M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
				tjj = math.Abs(ap.Get(ip - 1))
				if xj > tjj {
					xbnd = xbnd * (tjj / xj)
				}
				jlen = jlen + 1
				ip = ip + jinc*jlen
			}
			grow = minf64(grow, xbnd)
		} else {
			//           A is unit triangular.
			//
			//           Compute GROW = 1/G(j), where G(0) = max{x(i), i=1,...,n}.
			grow = minf64(one, one/maxf64(xbnd, smlnum))
			for _, j = range iter {
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
		err = goblas.Dtpsv(mat.UploByte(uplo), mat.TransByte(trans), mat.DiagByte(diag), *n, ap, x, 1)
	} else {
		//        Use a Level 1 BLAS solve, scaling intermediate results.
		if xmax > bignum {
			//           Scale X so that its components are less than or equal to
			//           BIGNUM in absolute value.
			(*scale) = bignum / xmax
			goblas.Dscal(*n, *scale, x, 1)
			xmax = bignum
		}

		if notran {
			//           Solve A * x = b
			ip = jfirst * (jfirst + 1) / 2
			for _, j = range iter {
				//              Compute x(j) = b(j) / A(j,j), scaling x if necessary.
				xj = math.Abs(x.Get(j - 1))
				if nounit {
					tjjs = ap.Get(ip-1) * tscal
				} else {
					tjjs = tscal
					if tscal == one {
						goto label100
					}
				}
				tjj = math.Abs(tjjs)
				if tjj > smlnum {
					//                    abs(A(j,j)) > SMLNUM:
					if tjj < one {
						if xj > tjj*bignum {
							//                          Scale x by 1/b(j).
							rec = one / xj
							goblas.Dscal(*n, rec, x, 1)
							(*scale) = (*scale) * rec
							xmax = xmax * rec
						}
					}
					x.Set(j-1, x.Get(j-1)/tjjs)
					xj = math.Abs(x.Get(j - 1))
				} else if tjj > zero {
					//                    0 < abs(A(j,j)) <= SMLNUM:
					if xj > tjj*bignum {
						//                       Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM
						//                       to avoid overflow when dividing by A(j,j).
						rec = (tjj * bignum) / xj
						if cnorm.Get(j-1) > one {
							//                          Scale by 1/CNORM(j) to avoid overflow when
							//                          multiplying x(j) times column j.
							rec = rec / cnorm.Get(j-1)
						}
						goblas.Dscal(*n, rec, x, 1)
						(*scale) = (*scale) * rec
						xmax = xmax * rec
					}
					x.Set(j-1, x.Get(j-1)/tjjs)
					xj = math.Abs(x.Get(j - 1))
				} else {
					//
					//                    A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
					//                    scale = 0, and compute a solution to A*x = 0.
					//
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
						//                    Scale x by 1/(2*abs(x(j))).
						rec = rec * half
						goblas.Dscal(*n, rec, x, 1)
						(*scale) = (*scale) * rec
					}
				} else if xj*cnorm.Get(j-1) > (bignum - xmax) {
					//                 Scale x by 1/2.
					goblas.Dscal(*n, half, x, 1)
					(*scale) = (*scale) * half
				}

				if upper {
					if j > 1 {
						//                    Compute the update
						//                       x(1:j-1) := x(1:j-1) - x(j) * A(1:j-1,j)
						goblas.Daxpy(j-1, -x.Get(j-1)*tscal, ap.Off(ip-j+1-1), 1, x, 1)
						i = goblas.Idamax(j-1, x, 1)
						xmax = math.Abs(x.Get(i - 1))
					}
					ip = ip - j
				} else {
					if j < (*n) {
						//                    Compute the update
						//                       x(j+1:n) := x(j+1:n) - x(j) * A(j+1:n,j)
						goblas.Daxpy((*n)-j, -x.Get(j-1)*tscal, ap.Off(ip+1-1), 1, x.Off(j+1-1), 1)
						i = j + goblas.Idamax((*n)-j, x.Off(j+1-1), 1)
						xmax = math.Abs(x.Get(i - 1))
					}
					ip = ip + (*n) - j + 1
				}
			}

		} else {
			//           Solve A**T * x = b
			ip = jfirst * (jfirst + 1) / 2
			jlen = 1
			for _, j = range iter {
				//              Compute x(j) = b(j) - sum A(k,j)*x(k).
				//                                    k<>j
				xj = math.Abs(x.Get(j - 1))
				uscal = tscal
				rec = one / maxf64(xmax, one)
				if cnorm.Get(j-1) > (bignum-xj)*rec {
					//                 If x(j) could overflow, scale x by 1/(2*XMAX).
					rec = rec * half
					if nounit {
						tjjs = ap.Get(ip-1) * tscal
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
						goblas.Dscal(*n, rec, x, 1)
						(*scale) = (*scale) * rec
						xmax = xmax * rec
					}
				}

				sumj = zero
				if uscal == one {
					//                 If the scaling needed for A in the dot product is 1,
					//                 call DDOT to perform the dot product.
					if upper {
						sumj = goblas.Ddot(j-1, ap.Off(ip-j+1-1), 1, x, 1)
					} else if j < (*n) {
						sumj = goblas.Ddot((*n)-j, ap.Off(ip+1-1), 1, x.Off(j+1-1), 1)
					}
				} else {
					//                 Otherwise, use in-line code for the dot product.
					if upper {
						for i = 1; i <= j-1; i++ {
							sumj = sumj + (ap.Get(ip-j+i-1)*uscal)*x.Get(i-1)
						}
					} else if j < (*n) {
						for i = 1; i <= (*n)-j; i++ {
							sumj = sumj + (ap.Get(ip+i-1)*uscal)*x.Get(j+i-1)
						}
					}
				}

				if uscal == tscal {
					//                 Compute x(j) := ( x(j) - sumj ) / A(j,j) if 1/A(j,j)
					//                 was not used to scale the dotproduct.
					x.Set(j-1, x.Get(j-1)-sumj)
					xj = math.Abs(x.Get(j - 1))
					if nounit {
						//                    Compute x(j) = x(j) / A(j,j), scaling if necessary.
						tjjs = ap.Get(ip-1) * tscal
					} else {
						tjjs = tscal
						if tscal == one {
							goto label150
						}
					}
					tjj = math.Abs(tjjs)
					if tjj > smlnum {
						//                       abs(A(j,j)) > SMLNUM:
						if tjj < one {
							if xj > tjj*bignum {
								//                             Scale X by 1/abs(x(j)).
								rec = one / xj
								goblas.Dscal(*n, rec, x, 1)
								(*scale) = (*scale) * rec
								xmax = xmax * rec
							}
						}
						x.Set(j-1, x.Get(j-1)/tjjs)
					} else if tjj > zero {
						//                       0 < abs(A(j,j)) <= SMLNUM:
						if xj > tjj*bignum {
							//                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
							rec = (tjj * bignum) / xj
							goblas.Dscal(*n, rec, x, 1)
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
				jlen = jlen + 1
				ip = ip + jinc*jlen
			}
		}
		(*scale) = (*scale) / tscal
	}

	//     Scale the column norms by 1/TSCAL for return.
	if tscal != one {
		goblas.Dscal(*n, one/tscal, cnorm, 1)
	}
}
