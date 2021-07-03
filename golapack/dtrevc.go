package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrevc computes some or all of the right and/or left eigenvectors of
// a real upper quasi-triangular matrix T.
// Matrices of this type are produced by the Schur factorization of
// a real general matrix:  A = Q*T*Q**T, as computed by DHSEQR.
//
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
//
//    T*x = w*x,     (y**H)*T = w*(y**H)
//
// where y**H denotes the conjugate transpose of y.
// The eigenvalues are not input to this routine, but are read directly
// from the diagonal blocks of T.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
// input matrix.  If Q is the orthogonal factor that reduces a matrix
// A to Schur form T, then Q*X and Q*Y are the matrices of right and
// left eigenvectors of A.
func Dtrevc(side, howmny byte, _select *[]bool, n *int, t *mat.Matrix, ldt *int, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr, mm, m *int, work *mat.Vector, info *int) {
	var allv, bothv, leftv, over, pair, rightv, somev bool
	var beta, bignum, emax, one, ovfl, rec, remax, scale, smin, smlnum, ulp, unfl, vcrit, vmax, wi, wr, xnorm, zero float64
	var i, ierr, ii, ip, is, j, j1, j2, jnxt, k, ki, n2 int
	var err error
	_ = err

	x := mf(2, 2, opts)

	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters
	bothv = side == 'B'
	rightv = side == 'R' || bothv
	leftv = side == 'L' || bothv

	allv = howmny == 'A'
	over = howmny == 'B'
	somev = howmny == 'S'

	(*info) = 0
	if !rightv && !leftv {
		(*info) = -1
	} else if !allv && !over && !somev {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvl) < 1 || (leftv && (*ldvl) < (*n)) {
		(*info) = -8
	} else if (*ldvr) < 1 || (rightv && (*ldvr) < (*n)) {
		(*info) = -10
	} else {
		//        Set M to the number of columns required to store the selected
		//        eigenvectors, standardize the array SELECT if necessary, and
		//        test MM.
		if somev {
			(*m) = 0
			pair = false
			for j = 1; j <= (*n); j++ {
				if pair {
					pair = false
					(*_select)[j-1] = false
				} else {
					if j < (*n) {
						if t.Get(j+1-1, j-1) == zero {
							if (*_select)[j-1] {
								(*m) = (*m) + 1
							}
						} else {
							pair = true
							if (*_select)[j-1] || (*_select)[j+1-1] {
								(*_select)[j-1] = true
								(*m) = (*m) + 2
							}
						}
					} else {
						if (*_select)[(*n)-1] {
							(*m) = (*m) + 1
						}
					}
				}
			}
		} else {
			(*m) = (*n)
		}

		if (*mm) < (*m) {
			(*info) = -11
		}
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DTREVC"), -(*info))
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	//     Set the constants to control overflow.
	unfl = Dlamch(SafeMinimum)
	ovfl = one / unfl
	Dlabad(&unfl, &ovfl)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(*n) / ulp)
	bignum = (one - ulp) / smlnum

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	work.Set(0, zero)
	for j = 2; j <= (*n); j++ {
		work.Set(j-1, zero)
		for i = 1; i <= j-1; i++ {
			work.Set(j-1, work.Get(j-1)+math.Abs(t.Get(i-1, j-1)))
		}
	}

	//     Index IP is used to specify the real or complex eigenvalue:
	//       IP = 0, real eigenvalue,
	//            1, first of conjugate complex pair: (wr,wi)
	//           -1, second of conjugate complex pair: (wr,wi)
	n2 = 2 * (*n)

	if rightv {
		//        Compute right eigenvectors.
		ip = 0
		is = (*m)
		for ki = (*n); ki >= 1; ki-- {

			if ip == 1 {
				goto label130
			}
			if ki == 1 {
				goto label40
			}
			if t.Get(ki-1, ki-1-1) == zero {
				goto label40
			}
			ip = -1

		label40:
			;
			if somev {
				if ip == 0 {
					if !(*_select)[ki-1] {
						goto label130
					}
				} else {
					if !(*_select)[ki-1-1] {
						goto label130
					}
				}
			}

			//           Compute the KI-th eigenvalue (WR,WI).
			wr = t.Get(ki-1, ki-1)
			wi = zero
			if ip != 0 {
				wi = math.Sqrt(math.Abs(t.Get(ki-1, ki-1-1))) * math.Sqrt(math.Abs(t.Get(ki-1-1, ki-1)))
			}
			smin = maxf64(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

			if ip == 0 {
				//              Real right eigenvector
				work.Set(ki+(*n)-1, one)

				//              Form right-hand side
				for k = 1; k <= ki-1; k++ {
					work.Set(k+(*n)-1, -t.Get(k-1, ki-1))
				}

				//              Solve the upper quasi-triangular system:
				//                 (T(1:KI-1,1:KI-1) - WR)*X = SCALE*WORK.
				jnxt = ki - 1
				for j = ki - 1; j >= 1; j-- {
					if j > jnxt {
						goto label60
					}
					j1 = j
					j2 = j
					jnxt = j - 1
					if j > 1 {
						if t.Get(j-1, j-1-1) != zero {
							j1 = j - 1
							jnxt = j - 2
						}
					}

					if j1 == j2 {
						//                    1-by-1 diagonal block
						Dlaln2(false, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, &zero, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale X(1,1) to avoid overflow when updating
						//                    the right-hand side.
						if xnorm > one {
							if work.Get(j-1) > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal(ki, scale, work.Off(1+(*n)-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))

						//                    Update right-hand side
						goblas.Daxpy(j-1, -x.Get(0, 0), t.Vector(0, j-1), 1, work.Off(1+(*n)-1), 1)

					} else {
						//                    2-by-2 diagonal block
						Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j-1-1, j-1-1), ldt, &one, &one, work.MatrixOff(j-1+(*n)-1, *n, opts), n, &wr, &zero, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale X(1,1) and X(2,1) to avoid overflow when
						//                    updating the right-hand side.
						if xnorm > one {
							beta = maxf64(work.Get(j-1-1), work.Get(j-1))
							if beta > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								x.Set(1, 0, x.Get(1, 0)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal(ki, scale, work.Off(1+(*n)-1), 1)
						}
						work.Set(j-1+(*n)-1, x.Get(0, 0))
						work.Set(j+(*n)-1, x.Get(1, 0))

						//                    Update right-hand side
						goblas.Daxpy(j-2, -x.Get(0, 0), t.Vector(0, j-1-1), 1, work.Off(1+(*n)-1), 1)
						goblas.Daxpy(j-2, -x.Get(1, 0), t.Vector(0, j-1), 1, work.Off(1+(*n)-1), 1)
					}
				label60:
				}

				//              Copy the vector x or Q*x to VR and normalize.
				if !over {
					goblas.Dcopy(ki, work.Off(1+(*n)-1), 1, vr.Vector(0, is-1), 1)

					ii = goblas.Idamax(ki, vr.Vector(0, is-1), 1)
					remax = one / math.Abs(vr.Get(ii-1, is-1))
					goblas.Dscal(ki, remax, vr.Vector(0, is-1), 1)

					for k = ki + 1; k <= (*n); k++ {
						vr.Set(k-1, is-1, zero)
					}
				} else {
					if ki > 1 {
						err = goblas.Dgemv(NoTrans, *n, ki-1, one, vr, *ldvr, work.Off(1+(*n)-1), 1, work.Get(ki+(*n)-1), vr.Vector(0, ki-1), 1)
					}

					ii = goblas.Idamax(*n, vr.Vector(0, ki-1), 1)
					remax = one / math.Abs(vr.Get(ii-1, ki-1))
					goblas.Dscal(*n, remax, vr.Vector(0, ki-1), 1)
				}

			} else {
				//              Complex right eigenvector.
				//
				//              Initial solve
				//                [ (T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I* WI)]*X = 0.
				//                [ (T(KI,KI-1)   T(KI,KI)   )               ]
				if math.Abs(t.Get(ki-1-1, ki-1)) >= math.Abs(t.Get(ki-1, ki-1-1)) {
					work.Set(ki-1+(*n)-1, one)
					work.Set(ki+n2-1, wi/t.Get(ki-1-1, ki-1))
				} else {
					work.Set(ki-1+(*n)-1, -wi/t.Get(ki-1, ki-1-1))
					work.Set(ki+n2-1, one)
				}
				work.Set(ki+(*n)-1, zero)
				work.Set(ki-1+n2-1, zero)

				//              Form right-hand side
				for k = 1; k <= ki-2; k++ {
					work.Set(k+(*n)-1, -work.Get(ki-1+(*n)-1)*t.Get(k-1, ki-1-1))
					work.Set(k+n2-1, -work.Get(ki+n2-1)*t.Get(k-1, ki-1))
				}

				//              Solve upper quasi-triangular system:
				//              (T(1:KI-2,1:KI-2) - (WR+i*WI))*X = SCALE*(WORK+i*WORK2)
				jnxt = ki - 2
				for j = ki - 2; j >= 1; j-- {
					if j > jnxt {
						goto label90
					}
					j1 = j
					j2 = j
					jnxt = j - 1
					if j > 1 {
						if t.Get(j-1, j-1-1) != zero {
							j1 = j - 1
							jnxt = j - 2
						}
					}

					if j1 == j2 {
						//                    1-by-1 diagonal block
						Dlaln2(false, func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale X(1,1) and X(1,2) to avoid overflow when
						//                    updating the right-hand side.
						if xnorm > one {
							if work.Get(j-1) > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								x.Set(0, 1, x.Get(0, 1)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal(ki, scale, work.Off(1+(*n)-1), 1)
							goblas.Dscal(ki, scale, work.Off(1+n2-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))
						work.Set(j+n2-1, x.Get(0, 1))

						//                    Update the right-hand side
						goblas.Daxpy(j-1, -x.Get(0, 0), t.Vector(0, j-1), 1, work.Off(1+(*n)-1), 1)
						goblas.Daxpy(j-1, -x.Get(0, 1), t.Vector(0, j-1), 1, work.Off(1+n2-1), 1)

					} else {
						//                    2-by-2 diagonal block
						Dlaln2(false, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &smin, &one, t.Off(j-1-1, j-1-1), ldt, &one, &one, work.MatrixOff(j-1+(*n)-1, *n, opts), n, &wr, &wi, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale X to avoid overflow when updating
						//                    the right-hand side.
						if xnorm > one {
							beta = maxf64(work.Get(j-1-1), work.Get(j-1))
							if beta > bignum/xnorm {
								rec = one / xnorm
								x.Set(0, 0, x.Get(0, 0)*rec)
								x.Set(0, 1, x.Get(0, 1)*rec)
								x.Set(1, 0, x.Get(1, 0)*rec)
								x.Set(1, 1, x.Get(1, 1)*rec)
								scale = scale * rec
							}
						}

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal(ki, scale, work.Off(1+(*n)-1), 1)
							goblas.Dscal(ki, scale, work.Off(1+n2-1), 1)
						}
						work.Set(j-1+(*n)-1, x.Get(0, 0))
						work.Set(j+(*n)-1, x.Get(1, 0))
						work.Set(j-1+n2-1, x.Get(0, 1))
						work.Set(j+n2-1, x.Get(1, 1))

						//                    Update the right-hand side
						goblas.Daxpy(j-2, -x.Get(0, 0), t.Vector(0, j-1-1), 1, work.Off(1+(*n)-1), 1)
						goblas.Daxpy(j-2, -x.Get(1, 0), t.Vector(0, j-1), 1, work.Off(1+(*n)-1), 1)
						goblas.Daxpy(j-2, -x.Get(0, 1), t.Vector(0, j-1-1), 1, work.Off(1+n2-1), 1)
						goblas.Daxpy(j-2, -x.Get(1, 1), t.Vector(0, j-1), 1, work.Off(1+n2-1), 1)
					}
				label90:
				}

				//              Copy the vector x or Q*x to VR and normalize.
				if !over {
					goblas.Dcopy(ki, work.Off(1+(*n)-1), 1, vr.Vector(0, is-1-1), 1)
					goblas.Dcopy(ki, work.Off(1+n2-1), 1, vr.Vector(0, is-1), 1)

					emax = zero
					for k = 1; k <= ki; k++ {
						emax = maxf64(emax, math.Abs(vr.Get(k-1, is-1-1))+math.Abs(vr.Get(k-1, is-1)))
					}

					remax = one / emax
					goblas.Dscal(ki, remax, vr.Vector(0, is-1-1), 1)
					goblas.Dscal(ki, remax, vr.Vector(0, is-1), 1)

					for k = ki + 1; k <= (*n); k++ {
						vr.Set(k-1, is-1-1, zero)
						vr.Set(k-1, is-1, zero)
					}

				} else {

					if ki > 2 {
						err = goblas.Dgemv(NoTrans, *n, ki-2, one, vr, *ldvr, work.Off(1+(*n)-1), 1, work.Get(ki-1+(*n)-1), vr.Vector(0, ki-1-1), 1)
						err = goblas.Dgemv(NoTrans, *n, ki-2, one, vr, *ldvr, work.Off(1+n2-1), 1, work.Get(ki+n2-1), vr.Vector(0, ki-1), 1)
					} else {
						goblas.Dscal(*n, work.Get(ki-1+(*n)-1), vr.Vector(0, ki-1-1), 1)
						goblas.Dscal(*n, work.Get(ki+n2-1), vr.Vector(0, ki-1), 1)
					}

					emax = zero
					for k = 1; k <= (*n); k++ {
						emax = maxf64(emax, math.Abs(vr.Get(k-1, ki-1-1))+math.Abs(vr.Get(k-1, ki-1)))
					}
					remax = one / emax
					goblas.Dscal(*n, remax, vr.Vector(0, ki-1-1), 1)
					goblas.Dscal(*n, remax, vr.Vector(0, ki-1), 1)
				}
			}

			is = is - 1
			if ip != 0 {
				is = is - 1
			}
		label130:
			;
			if ip == 1 {
				ip = 0
			}
			if ip == -1 {
				ip = 1
			}
		}
	}

	if leftv {
		//        Compute left eigenvectors.
		ip = 0
		is = 1
		for ki = 1; ki <= (*n); ki++ {

			if ip == -1 {
				goto label250
			}
			if ki == (*n) {
				goto label150
			}
			if t.Get(ki+1-1, ki-1) == zero {
				goto label150
			}
			ip = 1

		label150:
			;
			if somev {
				if !(*_select)[ki-1] {
					goto label250
				}
			}

			//           Compute the KI-th eigenvalue (WR,WI).
			wr = t.Get(ki-1, ki-1)
			wi = zero
			if ip != 0 {
				wi = math.Sqrt(math.Abs(t.Get(ki-1, ki+1-1))) * math.Sqrt(math.Abs(t.Get(ki+1-1, ki-1)))
			}
			smin = maxf64(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

			if ip == 0 {
				//              Real left eigenvector.
				work.Set(ki+(*n)-1, one)

				//              Form right-hand side
				for k = ki + 1; k <= (*n); k++ {
					work.Set(k+(*n)-1, -t.Get(ki-1, k-1))
				}

				//              Solve the quasi-triangular system:
				//                 (T(KI+1:N,KI+1:N) - WR)**T*X = SCALE*WORK
				vmax = one
				vcrit = bignum

				jnxt = ki + 1
				for j = ki + 1; j <= (*n); j++ {
					if j < jnxt {
						goto label170
					}
					j1 = j
					j2 = j
					jnxt = j + 1
					if j < (*n) {
						if t.Get(j+1-1, j-1) != zero {
							j2 = j + 1
							jnxt = j + 2
						}
					}

					if j1 == j2 {
						//                    1-by-1 diagonal block
						//
						//                    Scale if necessary to avoid overflow when forming
						//                    the right-hand side.
						if work.Get(j-1) > vcrit {
							rec = one / vmax
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+(*n)-1), 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+(*n)-1, work.Get(j+(*n)-1)-goblas.Ddot(j-ki-1, t.Vector(ki+1-1, j-1), 1, work.Off(ki+1+(*n)-1), 1))

						//                    Solve (T(J,J)-WR)**T*X = WORK
						Dlaln2(false, func() *int { y := 1; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, &zero, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+(*n)-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))
						vmax = maxf64(math.Abs(work.Get(j+(*n)-1)), vmax)
						vcrit = bignum / vmax

					} else {
						//                    2-by-2 diagonal block
						//
						//                    Scale if necessary to avoid overflow when forming
						//                    the right-hand side.
						beta = maxf64(work.Get(j-1), work.Get(j+1-1))
						if beta > vcrit {
							rec = one / vmax
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+(*n)-1), 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+(*n)-1, work.Get(j+(*n)-1)-goblas.Ddot(j-ki-1, t.Vector(ki+1-1, j-1), 1, work.Off(ki+1+(*n)-1), 1))

						work.Set(j+1+(*n)-1, work.Get(j+1+(*n)-1)-goblas.Ddot(j-ki-1, t.Vector(ki+1-1, j+1-1), 1, work.Off(ki+1+(*n)-1), 1))

						//                    Solve
						//                      [T(J,J)-WR   T(J,J+1)     ]**T * X = SCALE*( WORK1 )
						//                      [T(J+1,J)    T(J+1,J+1)-WR]                ( WORK2 )
						Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 1; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, &zero, x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale if necessary
						//
						if scale != one {
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+(*n)-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))
						work.Set(j+1+(*n)-1, x.Get(1, 0))
						//
						vmax = maxf64(math.Abs(work.Get(j+(*n)-1)), math.Abs(work.Get(j+1+(*n)-1)), vmax)
						vcrit = bignum / vmax

					}
				label170:
				}

				//              Copy the vector x or Q*x to VL and normalize.
				if !over {
					goblas.Dcopy((*n)-ki+1, work.Off(ki+(*n)-1), 1, vl.Vector(ki-1, is-1), 1)

					ii = goblas.Idamax((*n)-ki+1, vl.Vector(ki-1, is-1), 1) + ki - 1
					remax = one / math.Abs(vl.Get(ii-1, is-1))
					goblas.Dscal((*n)-ki+1, remax, vl.Vector(ki-1, is-1), 1)

					for k = 1; k <= ki-1; k++ {
						vl.Set(k-1, is-1, zero)
					}

				} else {

					if ki < (*n) {
						err = goblas.Dgemv(NoTrans, *n, (*n)-ki, one, vl.Off(0, ki+1-1), *ldvl, work.Off(ki+1+(*n)-1), 1, work.Get(ki+(*n)-1), vl.Vector(0, ki-1), 1)
					}

					ii = goblas.Idamax(*n, vl.Vector(0, ki-1), 1)
					remax = one / math.Abs(vl.Get(ii-1, ki-1))
					goblas.Dscal(*n, remax, vl.Vector(0, ki-1), 1)

				}

			} else {
				//              Complex left eigenvector.
				//
				//               Initial solve:
				//                 ((T(KI,KI)    T(KI,KI+1) )**T - (WR - I* WI))*X = 0.
				//                 ((T(KI+1,KI) T(KI+1,KI+1))                )
				if math.Abs(t.Get(ki-1, ki+1-1)) >= math.Abs(t.Get(ki+1-1, ki-1)) {
					work.Set(ki+(*n)-1, wi/t.Get(ki-1, ki+1-1))
					work.Set(ki+1+n2-1, one)
				} else {
					work.Set(ki+(*n)-1, one)
					work.Set(ki+1+n2-1, -wi/t.Get(ki+1-1, ki-1))
				}
				work.Set(ki+1+(*n)-1, zero)
				work.Set(ki+n2-1, zero)

				//              Form right-hand side
				for k = ki + 2; k <= (*n); k++ {
					work.Set(k+(*n)-1, -work.Get(ki+(*n)-1)*t.Get(ki-1, k-1))
					work.Set(k+n2-1, -work.Get(ki+1+n2-1)*t.Get(ki+1-1, k-1))
				}

				//              Solve complex quasi-triangular system:
				//              ( T(KI+2,N:KI+2,N) - (WR-i*WI) )*X = WORK1+i*WORK2
				vmax = one
				vcrit = bignum

				jnxt = ki + 2
				for j = ki + 2; j <= (*n); j++ {
					if j < jnxt {
						goto label200
					}
					j1 = j
					j2 = j
					jnxt = j + 1
					if j < (*n) {
						if t.Get(j+1-1, j-1) != zero {
							j2 = j + 1
							jnxt = j + 2
						}
					}

					if j1 == j2 {
						//                    1-by-1 diagonal block
						//
						//                    Scale if necessary to avoid overflow when
						//                    forming the right-hand side elements.
						if work.Get(j-1) > vcrit {
							rec = one / vmax
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+(*n)-1), 1)
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+n2-1), 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+(*n)-1, work.Get(j+(*n)-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j-1), 1, work.Off(ki+2+(*n)-1), 1))
						work.Set(j+n2-1, work.Get(j+n2-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j-1), 1, work.Off(ki+2+n2-1), 1))

						//                    Solve (T(J,J)-(WR-i*WI))*(X11+i*X12)= WK+I*WK2
						Dlaln2(false, func() *int { y := 1; return &y }(), func() *int { y := 2; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, func() *float64 { y := -wi; return &y }(), x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+(*n)-1), 1)
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+n2-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))
						work.Set(j+n2-1, x.Get(0, 1))
						vmax = maxf64(math.Abs(work.Get(j+(*n)-1)), math.Abs(work.Get(j+n2-1)), vmax)
						vcrit = bignum / vmax

					} else {
						//                    2-by-2 diagonal block
						//
						//                    Scale if necessary to avoid overflow when forming
						//                    the right-hand side elements.
						beta = maxf64(work.Get(j-1), work.Get(j+1-1))
						if beta > vcrit {
							rec = one / vmax
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+(*n)-1), 1)
							goblas.Dscal((*n)-ki+1, rec, work.Off(ki+n2-1), 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+(*n)-1, work.Get(j+(*n)-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j-1), 1, work.Off(ki+2+(*n)-1), 1))

						work.Set(j+n2-1, work.Get(j+n2-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j-1), 1, work.Off(ki+2+n2-1), 1))

						work.Set(j+1+(*n)-1, work.Get(j+1+(*n)-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j+1-1), 1, work.Off(ki+2+(*n)-1), 1))

						work.Set(j+1+n2-1, work.Get(j+1+n2-1)-goblas.Ddot(j-ki-2, t.Vector(ki+2-1, j+1-1), 1, work.Off(ki+2+n2-1), 1))

						//                    Solve 2-by-2 complex linear equation
						//                      ([T(j,j)   T(j,j+1)  ]**T-(wr-i*wi)*I)*X = SCALE*B
						//                      ([T(j+1,j) T(j+1,j+1)]               )
						Dlaln2(true, func() *int { y := 2; return &y }(), func() *int { y := 2; return &y }(), &smin, &one, t.Off(j-1, j-1), ldt, &one, &one, work.MatrixOff(j+(*n)-1, *n, opts), n, &wr, func() *float64 { y := -wi; return &y }(), x, func() *int { y := 2; return &y }(), &scale, &xnorm, &ierr)

						//                    Scale if necessary
						if scale != one {
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+(*n)-1), 1)
							goblas.Dscal((*n)-ki+1, scale, work.Off(ki+n2-1), 1)
						}
						work.Set(j+(*n)-1, x.Get(0, 0))
						work.Set(j+n2-1, x.Get(0, 1))
						work.Set(j+1+(*n)-1, x.Get(1, 0))
						work.Set(j+1+n2-1, x.Get(1, 1))
						vmax = maxf64(math.Abs(x.Get(0, 0)), math.Abs(x.Get(0, 1)), math.Abs(x.Get(1, 0)), math.Abs(x.Get(1, 1)), vmax)
						vcrit = bignum / vmax

					}
				label200:
				}

				//              Copy the vector x or Q*x to VL and normalize.
				if !over {
					goblas.Dcopy((*n)-ki+1, work.Off(ki+(*n)-1), 1, vl.Vector(ki-1, is-1), 1)
					goblas.Dcopy((*n)-ki+1, work.Off(ki+n2-1), 1, vl.Vector(ki-1, is+1-1), 1)
					//
					emax = zero
					for k = ki; k <= (*n); k++ {
						emax = maxf64(emax, math.Abs(vl.Get(k-1, is-1))+math.Abs(vl.Get(k-1, is+1-1)))
					}
					remax = one / emax
					goblas.Dscal((*n)-ki+1, remax, vl.Vector(ki-1, is-1), 1)
					goblas.Dscal((*n)-ki+1, remax, vl.Vector(ki-1, is+1-1), 1)

					for k = 1; k <= ki-1; k++ {
						vl.Set(k-1, is-1, zero)
						vl.Set(k-1, is+1-1, zero)
					}
				} else {
					if ki < (*n)-1 {
						err = goblas.Dgemv(NoTrans, *n, (*n)-ki-1, one, vl.Off(0, ki+2-1), *ldvl, work.Off(ki+2+(*n)-1), 1, work.Get(ki+(*n)-1), vl.Vector(0, ki-1), 1)
						err = goblas.Dgemv(NoTrans, *n, (*n)-ki-1, one, vl.Off(0, ki+2-1), *ldvl, work.Off(ki+2+n2-1), 1, work.Get(ki+1+n2-1), vl.Vector(0, ki+1-1), 1)
					} else {
						goblas.Dscal(*n, work.Get(ki+(*n)-1), vl.Vector(0, ki-1), 1)
						goblas.Dscal(*n, work.Get(ki+1+n2-1), vl.Vector(0, ki+1-1), 1)
					}

					emax = zero
					for k = 1; k <= (*n); k++ {
						emax = maxf64(emax, math.Abs(vl.Get(k-1, ki-1))+math.Abs(vl.Get(k-1, ki+1-1)))
					}
					remax = one / emax
					goblas.Dscal(*n, remax, vl.Vector(0, ki-1), 1)
					goblas.Dscal(*n, remax, vl.Vector(0, ki+1-1), 1)

				}

			}

			is = is + 1
			if ip != 0 {
				is = is + 1
			}
		label250:
			;
			if ip == -1 {
				ip = 0
			}
			if ip == 1 {
				ip = -1
			}

		}

	}
}
