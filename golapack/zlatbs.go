package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zlatbs solves one of the triangular systems
//
//    A * x = s*b,  A**T * x = s*b,  or  A**H * x = s*b,
//
// with scaling to prevent overflow, where A is an upper or lower
// triangular band matrix.  Here A**T denotes the transpose of A, x and b
// are n-element vectors, and s is a scaling factor, usually less than
// or equal to 1, chosen so that the components of x will be less than
// the overflow threshold.  If the unscaled problem will not cause
// overflow, the Level 2 BLAS routine ZTBSV is called.  If the matrix A
// is singular (A(j,j) = 0 for some j), then s is set to 0 and a
// non-trivial solution to A*x = 0 is returned.
func Zlatbs(uplo mat.MatUplo, trans mat.MatTrans, diag mat.MatDiag, normin byte, n, kd int, ab *mat.CMatrix, x *mat.CVector, cnorm *mat.Vector) (scale float64, err error) {
	var notran, nounit, upper bool
	var csumj, tjjs, uscal complex128
	var bignum, grow, half, one, rec, smlnum, tjj, tmax, tscal, two, xbnd, xj, xmax, zero float64
	var i, imax, j, jfirst, jinc, jlast, jlen, maind int

	zero = 0.0
	half = 0.5
	one = 1.0
	two = 2.0

	upper = uplo == Upper
	notran = trans == NoTrans
	nounit = diag == NonUnit

	//     Test the input parameters.
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if !notran && trans != Trans && trans != ConjTrans {
		err = fmt.Errorf("!notran && trans != Trans && trans != ConjTrans: trans=%s", trans)
	} else if !nounit && diag != Unit {
		err = fmt.Errorf("!nounit && diag != Unit: diag=%s", diag)
	} else if normin != 'Y' && normin != 'N' {
		err = fmt.Errorf("normin != 'Y' && normin != 'N': normin='%c'", normin)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if kd < 0 {
		err = fmt.Errorf("kd < 0: kd=%v", kd)
	} else if ab.Rows < kd+1 {
		err = fmt.Errorf("ab.Rows < kd+1: ab.Rows=%v, kd=%v", ab.Rows, kd)
	}
	if err != nil {
		gltest.Xerbla2("Zlatbs", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Determine machine dependent parameters to control overflow.
	smlnum = Dlamch(SafeMinimum)
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)
	smlnum = smlnum / Dlamch(Precision)
	bignum = one / smlnum
	scale = one

	if normin == 'N' {
		//        Compute the 1-norm of each column, not including the diagonal.
		if upper {
			//           A is upper triangular.
			for j = 1; j <= n; j++ {
				jlen = min(kd, j-1)
				cnorm.Set(j-1, ab.Off(kd+1-jlen-1, j-1).CVector().Asum(jlen, 1))
			}
		} else {
			//           A is lower triangular.
			for j = 1; j <= n; j++ {
				jlen = min(kd, n-j)
				if jlen > 0 {
					cnorm.Set(j-1, ab.Off(1, j-1).CVector().Asum(jlen, 1))
				} else {
					cnorm.Set(j-1, zero)
				}
			}
		}
	}

	//     Scale the column norms by TSCAL if the maximum element in CNORM is
	//     greater than BIGNUM/2.
	imax = cnorm.Iamax(n, 1)
	tmax = cnorm.Get(imax - 1)
	if tmax <= bignum*half {
		tscal = one
	} else {
		tscal = half / (smlnum * tmax)
		cnorm.Scal(n, tscal, 1)
	}

	//     Compute a bound on the computed solution vector to see if the
	//     Level 2 BLAS routine ZTBSV can be used.
	xmax = zero
	for j = 1; j <= n; j++ {
		xmax = math.Max(xmax, cabs2(x.Get(j-1)))
	}
	xbnd = xmax
	if notran {
		//        Compute the growth in A * x = b.
		if upper {
			jfirst = n
			jlast = 1
			jinc = -1
			maind = kd + 1
		} else {
			jfirst = 1
			jlast = n
			jinc = 1
			maind = 1
		}

		if tscal != one {
			grow = zero
			goto label60
		}

		if nounit {
			//           A is non-unit triangular.
			//
			//           Compute GROW = 1/G(j) and XBND = 1/M(j).
			//           Initially, G(0) = math.Max{x(i), i=1,...,n}.
			grow = half / math.Max(xbnd, smlnum)
			xbnd = grow
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label60
				}

				tjjs = ab.Get(maind-1, j-1)
				tjj = cabs1(tjjs)

				if tjj >= smlnum {
					//                 M(j) = G(j-1) / abs(A(j,j))
					xbnd = math.Min(xbnd, math.Min(one, tjj)*grow)
				} else {
					//                 M(j) could overflow, set XBND to 0.
					xbnd = zero
				}

				if tjj+cnorm.Get(j-1) >= smlnum {
					//                 G(j) = G(j-1)*( 1 + CNORM(j) / abs(A(j,j)) )
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
			//           Compute GROW = 1/G(j), where G(0) = math.Max{x(i), i=1,...,n}.
			grow = math.Min(one, half/math.Max(xbnd, smlnum))
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label60
				}

				//              G(j) = G(j-1)*( 1 + CNORM(j) )
				grow = grow * (one / (one + cnorm.Get(j-1)))
			}
		}
	label60:
	} else {
		//        Compute the growth in A**T * x = b  or  A**H * x = b.
		if upper {
			jfirst = 1
			jlast = n
			jinc = 1
			maind = kd + 1
		} else {
			jfirst = n
			jlast = 1
			jinc = -1
			maind = 1
		}

		if tscal != one {
			grow = zero
			goto label90
		}

		if nounit {
			//           A is non-unit triangular.
			//
			//           Compute GROW = 1/G(j) and XBND = 1/M(j).
			//           Initially, M(0) = math.Max{x(i), i=1,...,n}.
			grow = half / math.Max(xbnd, smlnum)
			xbnd = grow
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label90
				}

				//              G(j) = math.Max( G(j-1), M(j-1)*( 1 + CNORM(j) ) )
				xj = one + cnorm.Get(j-1)
				grow = math.Min(grow, xbnd/xj)

				tjjs = ab.Get(maind-1, j-1)
				tjj = cabs1(tjjs)

				if tjj >= smlnum {
					//                 M(j) = M(j-1)*( 1 + CNORM(j) ) / abs(A(j,j))
					if xj > tjj {
						xbnd = xbnd * (tjj / xj)
					}
				} else {
					//                 M(j) could overflow, set XBND to 0.
					xbnd = zero
				}
			}
			grow = math.Min(grow, xbnd)
		} else {
			//           A is unit triangular.
			//
			//           Compute GROW = 1/G(j), where G(0) = math.Max{x(i), i=1,...,n}.
			grow = math.Min(one, half/math.Max(xbnd, smlnum))
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Exit the loop if the growth factor is too small.
				if grow <= smlnum {
					goto label90
				}

				//              G(j) = ( 1 + CNORM(j) )*G(j-1)
				xj = one + cnorm.Get(j-1)
				grow = grow / xj
			}
		}
	label90:
	}

	if (grow * tscal) > smlnum {
		//        Use the Level 2 BLAS solve if the reciprocal of the bound on
		//        elements of X is not too small.
		if err = x.Tbsv(uplo, trans, diag, n, kd, ab, 1); err != nil {
			panic(err)
		}
	} else {
		//        Use a Level 1 BLAS solve, scaling intermediate results.
		if xmax > bignum*half {
			//           Scale X so that its components are less than or equal to
			//           BIGNUM in absolute value.
			scale = (bignum * half) / xmax
			x.Dscal(n, scale, 1)
			xmax = bignum
		} else {
			xmax = xmax * two
		}

		if notran {
			//           Solve A * x = b
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Compute x(j) = b(j) / A(j,j), scaling x if necessary.
				xj = cabs1(x.Get(j - 1))
				if nounit {
					tjjs = ab.Get(maind-1, j-1) * complex(tscal, 0)
				} else {
					tjjs = complex(tscal, 0)
					if tscal == one {
						goto label110
					}
				}
				tjj = cabs1(tjjs)
				if tjj > smlnum {
					//                    abs(A(j,j)) > SMLNUM:
					if tjj < one {
						if xj > tjj*bignum {
							//                          Scale x by 1/b(j).
							rec = one / xj
							x.Dscal(n, rec, 1)
							scale = scale * rec
							xmax = xmax * rec
						}
					}
					x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					xj = cabs1(x.Get(j - 1))
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
						x.Dscal(n, rec, 1)
						scale = scale * rec
						xmax = xmax * rec
					}
					x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					xj = cabs1(x.Get(j - 1))
				} else {
					//                    A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
					//                    scale = 0, and compute a solution to A*x = 0.
					for i = 1; i <= n; i++ {
						x.SetRe(i-1, zero)
					}
					x.SetRe(j-1, one)
					xj = one
					scale = zero
					xmax = zero
				}
			label110:
				;

				//              Scale x if necessary to avoid overflow when adding a
				//              multiple of column j of A.
				if xj > one {
					rec = one / xj
					if cnorm.Get(j-1) > (bignum-xmax)*rec {
						//                    Scale x by 1/(2*abs(x(j))).
						rec = rec * half
						x.Dscal(n, rec, 1)
						scale = scale * rec
					}
				} else if xj*cnorm.Get(j-1) > (bignum - xmax) {
					//                 Scale x by 1/2.
					x.Dscal(n, half, 1)
					scale = scale * half
				}

				if upper {
					if j > 1 {
						//                    Compute the update
						//                       x(math.Max(1,j-kd):j-1) := x(math.Max(1,j-kd):j-1) -
						//                                             x(j)* A(math.Max(1,j-kd):j-1,j)
						jlen = min(kd, j-1)
						x.Off(j-jlen-1).Axpy(jlen, -x.Get(j-1)*complex(tscal, 0), ab.Off(kd+1-jlen-1, j-1).CVector(), 1, 1)
						i = x.Iamax(j-1, 1)
						xmax = cabs1(x.Get(i - 1))
					}
				} else if j < n {
					//                 Compute the update
					//                    x(j+1:min(j+kd,n)) := x(j+1:min(j+kd,n)) -
					//                                          x(j) * A(j+1:min(j+kd,n),j)
					jlen = min(kd, n-j)
					if jlen > 0 {
						x.Off(j).Axpy(jlen, -x.Get(j-1)*complex(tscal, 0), ab.Off(1, j-1).CVector(), 1, 1)
					}
					i = j + x.Off(j).Iamax(n-j, 1)
					xmax = cabs1(x.Get(i - 1))
				}
			}

		} else if trans == 'T' {
			//           Solve A**T * x = b
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Compute x(j) = b(j) - sum A(k,j)*x(k).
				//                                    k<>j
				xj = cabs1(x.Get(j - 1))
				uscal = complex(tscal, 0)
				rec = one / math.Max(xmax, one)
				if cnorm.Get(j-1) > (bignum-xj)*rec {
					//                 If x(j) could overflow, scale x by 1/(2*XMAX).
					rec = rec * half
					if nounit {
						tjjs = ab.Get(maind-1, j-1) * complex(tscal, 0)
					} else {
						tjjs = complex(tscal, 0)
					}
					tjj = cabs1(tjjs)
					if tjj > one {
						//                       Divide by A(j,j) when scaling x if A(j,j) > 1.
						rec = math.Min(one, rec*tjj)
						uscal = Zladiv(uscal, tjjs)
					}
					if rec < one {
						x.Dscal(n, rec, 1)
						scale = scale * rec
						xmax = xmax * rec
					}
				}

				csumj = complex(zero, 0)
				if uscal == complex(one, 0) {
					//                 If the scaling needed for A in the dot product is 1,
					//                 call ZDOTU to perform the dot product.
					if upper {
						jlen = min(kd, j-1)
						csumj = x.Off(j-jlen-1).Dotu(jlen, ab.Off(kd+1-jlen-1, j-1).CVector(), 1, 1)
					} else {
						jlen = min(kd, n-j)
						if jlen > 1 {
							csumj = x.Off(j).Dotu(jlen, ab.Off(1, j-1).CVector(), 1, 1)
						}
					}
				} else {
					//                 Otherwise, use in-line code for the dot product.
					if upper {
						jlen = min(kd, j-1)
						for i = 1; i <= jlen; i++ {
							csumj = csumj + (ab.Get(kd+i-jlen-1, j-1)*uscal)*x.Get(j-jlen-1+i-1)
						}
					} else {
						jlen = min(kd, n-j)
						for i = 1; i <= jlen; i++ {
							csumj = csumj + (ab.Get(i, j-1)*uscal)*x.Get(j+i-1)
						}
					}
				}

				if uscal == complex(tscal, 0) {
					//                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
					//                 was not used to scale the dotproduct.
					x.Set(j-1, x.Get(j-1)-csumj)
					xj = cabs1(x.Get(j - 1))
					if nounit {
						//                    Compute x(j) = x(j) / A(j,j), scaling if necessary.
						tjjs = ab.Get(maind-1, j-1) * complex(tscal, 0)
					} else {
						tjjs = complex(tscal, 0)
						if tscal == one {
							goto label160
						}
					}
					tjj = cabs1(tjjs)
					if tjj > smlnum {
						//                       abs(A(j,j)) > SMLNUM:
						if tjj < one {
							if xj > tjj*bignum {
								//                             Scale X by 1/abs(x(j)).
								rec = one / xj
								x.Dscal(n, rec, 1)
								scale = scale * rec
								xmax = xmax * rec
							}
						}
						x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					} else if tjj > zero {
						//                       0 < abs(A(j,j)) <= SMLNUM:
						if xj > tjj*bignum {
							//                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
							rec = (tjj * bignum) / xj
							x.Dscal(n, rec, 1)
							scale = scale * rec
							xmax = xmax * rec
						}
						x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					} else {
						//                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
						//                       scale = 0 and compute a solution to A**T *x = 0.
						for i = 1; i <= n; i++ {
							x.Set(i-1, complex(zero, 0))
						}
						x.Set(j-1, complex(one, 0))
						scale = zero
						xmax = zero
					}
				label160:
				} else {
					//                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
					//                 product has already been divided by 1/A(j,j).
					x.Set(j-1, Zladiv(x.Get(j-1), tjjs)-csumj)
				}
				xmax = math.Max(xmax, cabs1(x.Get(j-1)))
			}

		} else {
			//           Solve A**H * x = b
			for _, j = range genIter(jfirst, jlast, jinc) {
				//              Compute x(j) = b(j) - sum A(k,j)*x(k).
				//                                    k<>j
				xj = cabs1(x.Get(j - 1))
				uscal = complex(tscal, 0)
				rec = one / math.Max(xmax, one)
				if cnorm.Get(j-1) > (bignum-xj)*rec {
					//                 If x(j) could overflow, scale x by 1/(2*XMAX).
					rec = rec * half
					if nounit {
						tjjs = ab.GetConj(maind-1, j-1) * complex(tscal, 0)
					} else {
						tjjs = complex(tscal, 0)
					}
					tjj = cabs1(tjjs)
					if tjj > one {
						//                       Divide by A(j,j) when scaling x if A(j,j) > 1.
						rec = math.Min(one, rec*tjj)
						uscal = Zladiv(uscal, tjjs)
					}
					if rec < one {
						x.Dscal(n, rec, 1)
						scale = scale * rec
						xmax = xmax * rec
					}
				}

				csumj = complex(zero, 0)
				if uscal == complex(one, 0) {
					//                 If the scaling needed for A in the dot product is 1,
					//                 call ZDOTC to perform the dot product.
					if upper {
						jlen = min(kd, j-1)
						csumj = x.Off(j-jlen-1).Dotc(jlen, ab.Off(kd+1-jlen-1, j-1).CVector(), 1, 1)
					} else {
						jlen = min(kd, n-j)
						if jlen > 1 {
							csumj = x.Off(j).Dotc(jlen, ab.Off(1, j-1).CVector(), 1, 1)
						}
					}
				} else {
					//                 Otherwise, use in-line code for the dot product.
					if upper {
						jlen = min(kd, j-1)
						for i = 1; i <= jlen; i++ {
							csumj = csumj + (ab.GetConj(kd+i-jlen-1, j-1)*uscal)*x.Get(j-jlen-1+i-1)
						}
					} else {
						jlen = min(kd, n-j)
						for i = 1; i <= jlen; i++ {
							csumj = csumj + (ab.GetConj(i, j-1)*uscal)*x.Get(j+i-1)
						}
					}
				}

				if uscal == complex(tscal, 0) {
					//                 Compute x(j) := ( x(j) - CSUMJ ) / A(j,j) if 1/A(j,j)
					//                 was not used to scale the dotproduct.
					x.Set(j-1, x.Get(j-1)-csumj)
					xj = cabs1(x.Get(j - 1))
					if nounit {
						//                    Compute x(j) = x(j) / A(j,j), scaling if necessary.
						tjjs = ab.GetConj(maind-1, j-1) * complex(tscal, 0)
					} else {
						tjjs = complex(tscal, 0)
						if tscal == one {
							goto label210
						}
					}
					tjj = cabs1(tjjs)
					if tjj > smlnum {
						//                       abs(A(j,j)) > SMLNUM:
						if tjj < one {
							if xj > tjj*bignum {
								//                             Scale X by 1/abs(x(j)).
								rec = one / xj
								x.Dscal(n, rec, 1)
								scale = scale * rec
								xmax = xmax * rec
							}
						}
						x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					} else if tjj > zero {
						//                       0 < abs(A(j,j)) <= SMLNUM:
						if xj > tjj*bignum {
							//                          Scale x by (1/abs(x(j)))*abs(A(j,j))*BIGNUM.
							rec = (tjj * bignum) / xj
							x.Dscal(n, rec, 1)
							scale = scale * rec
							xmax = xmax * rec
						}
						x.Set(j-1, Zladiv(x.Get(j-1), tjjs))
					} else {
						//                       A(j,j) = 0:  Set x(1:n) = 0, x(j) = 1, and
						//                       scale = 0 and compute a solution to A**H *x = 0.
						for i = 1; i <= n; i++ {
							x.SetRe(i-1, zero)
						}
						x.SetRe(j-1, one)
						scale = zero
						xmax = zero
					}
				label210:
				} else {
					//                 Compute x(j) := x(j) / A(j,j) - CSUMJ if the dot
					//                 product has already been divided by 1/A(j,j).
					x.Set(j-1, Zladiv(x.Get(j-1), tjjs)-csumj)
				}
				xmax = math.Max(xmax, cabs1(x.Get(j-1)))
			}
		}
		scale = scale / tscal
	}

	//     Scale the column norms by 1/TSCAL for return.
	if tscal != one {
		cnorm.Scal(n, one/tscal, 1)
	}

	return
}
