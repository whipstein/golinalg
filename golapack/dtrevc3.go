package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dtrevc3 computes some or all of the right and/or left eigenvectors of
// a real upper quasi-triangular matrix T.
// Matrices of this type are produced by the Schur factorization of
// a real general matrix:  A = Q*T*Q**T, as computed by DHSEQR.
//
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
//
//    T*x = w*x,     (y**T)*T = w*(y**T)
//
// where y**T denotes the transpose of the vector y.
// The eigenvalues are not input to this routine, but are read directly
// from the diagonal blocks of T.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
// input matrix. If Q is the orthogonal factor that reduces a matrix
// A to Schur form T, then Q*X and Q*Y are the matrices of right and
// left eigenvectors of A.
//
// This uses a Level 3 BLAS version of the back transformation.
func Dtrevc3(side mat.MatSide, howmny byte, _select *[]bool, n int, t, vl, vr *mat.Matrix, mm int, work *mat.Vector, lwork int) (m int, err error) {
	var allv, bothv, leftv, lquery, over, pair, rightv, somev bool
	var beta, bignum, emax, one, ovfl, rec, remax, scale, smin, smlnum, ulp, unfl, vcrit, vmax, wi, wr, xnorm, zero float64
	var i, ii, ip, is, iv, j, j1, j2, jnxt, k, ki, ki2, maxwrk, nb, nbmax, nbmin int

	iscomplex := make([]int, 128)
	x := mf(2, 2, opts)

	zero = 0.0
	one = 1.0
	nbmin = 8
	nbmax = 128

	//     Decode and test the input parameters
	bothv = side == Both
	rightv = side == Right || bothv
	leftv = side == Left || bothv

	allv = howmny == 'A'
	over = howmny == 'B'
	somev = howmny == 'S'

	nb = Ilaenv(1, "Dtrevc", []byte{side.Byte(), howmny}, n, -1, -1, -1)
	maxwrk = n + 2*n*nb
	work.Set(0, float64(maxwrk))
	lquery = (lwork == -1)
	if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if !allv && !over && !somev {
		err = fmt.Errorf("!allv && !over && !somev: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if vl.Rows < 1 || (leftv && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=%s, vl.Rows=%v, n=%v", side, vl.Rows, n)
	} else if vr.Rows < 1 || (rightv && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=%s, vr.Rows=%v, n=%v", side, vr.Rows, n)
	} else if lwork < max(1, 3*n) && !lquery {
		err = fmt.Errorf("lwork < max(1, 3*n) && !lquery: lwork=%v, n=%v, lquery=%v", lwork, n, lquery)
	} else {
		//        Set M to the number of columns required to store the selected
		//        eigenvectors, standardize the array SELECT if necessary, and
		//        test MM.
		if somev {
			m = 0
			pair = false
			for j = 1; j <= n; j++ {
				if pair {
					pair = false
					(*_select)[j-1] = false
				} else {
					if j < n {
						if t.Get(j, j-1) == zero {
							if (*_select)[j-1] {
								m = m + 1
							}
						} else {
							pair = true
							if (*_select)[j-1] || (*_select)[j] {
								(*_select)[j-1] = true
								m = m + 2
							}
						}
					} else {
						if (*_select)[n-1] {
							m = m + 1
						}
					}
				}
			}
		} else {
			m = n
		}

		if mm < m {
			err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
		}
	}
	if err != nil {
		gltest.Xerbla2("Dtrevc3", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	//     Use blocked version of back-transformation if sufficient workspace.
	//     Zero-out the workspace to avoid potential NaN propagation.
	if over && lwork >= n+2*n*nbmin {
		nb = (lwork - n) / (2 * n)
		nb = min(nb, nbmax)
		Dlaset(Full, n, 1+2*nb, zero, zero, work.Matrix(n, opts))
	} else {
		nb = 1
	}

	//     Set the constants to control overflow.
	unfl = Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = Dlabad(unfl, ovfl)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(n) / ulp)
	bignum = (one - ulp) / smlnum

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	work.Set(0, zero)
	for j = 2; j <= n; j++ {
		work.Set(j-1, zero)
		for i = 1; i <= j-1; i++ {
			work.Set(j-1, work.Get(j-1)+math.Abs(t.Get(i-1, j-1)))
		}
	}

	//     Index IP is used to specify the real or complex eigenvalue:
	//       IP = 0, real eigenvalue,
	//            1, first  of conjugate complex pair: (wr,wi)
	//           -1, second of conjugate complex pair: (wr,wi)
	//       ISCOMPLEX array stores IP for each column in current block.
	if rightv {
		//        ============================================================
		//        Compute right eigenvectors.
		//
		//        IV is index of column in current block.
		//        For complex right vector, uses IV-1 for real part and IV for complex part.
		//        Non-blocked version always uses IV=2;
		//        blocked     version starts with IV=NB, goes down to 1 or 2.
		//        (Note the "0-th" column is used for 1-norms computed above.)
		iv = 2
		if nb > 2 {
			iv = nb
		}
		ip = 0
		is = m
		for ki = n; ki >= 1; ki-- {
			if ip == -1 {
				//              previous iteration (ki+1) was second of conjugate pair,
				//              so this ki is first of conjugate pair; skip to end of loop
				ip = 1
				goto label140
			} else if ki == 1 {
				//              last column, so this ki must be real eigenvalue
				ip = 0
			} else if t.Get(ki-1, ki-1-1) == zero {
				//              zero on sub-diagonal, so this ki is real eigenvalue
				ip = 0
			} else {
				//              non-zero on sub-diagonal, so this ki is second of conjugate pair
				ip = -1
			}
			if somev {
				if ip == 0 {
					if !(*_select)[ki-1] {
						goto label140
					}
				} else {
					if !(*_select)[ki-1-1] {
						goto label140
					}
				}
			}

			//           Compute the KI-th eigenvalue (WR,WI).
			wr = t.Get(ki-1, ki-1)
			wi = zero
			if ip != 0 {
				wi = math.Sqrt(math.Abs(t.Get(ki-1, ki-1-1))) * math.Sqrt(math.Abs(t.Get(ki-1-1, ki-1)))
			}
			smin = math.Max(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

			if ip == 0 {
				//              --------------------------------------------------------
				//              Real right eigenvector
				work.Set(ki+iv*n-1, one)

				//              Form right-hand side.
				for k = 1; k <= ki-1; k++ {
					work.Set(k+iv*n-1, -t.Get(k-1, ki-1))
				}

				//              Solve upper quasi-triangular system:
				//              [ T(1:KI-1,1:KI-1) - WR ]*X = SCALE*WORK.
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
						if scale, xnorm, _ = Dlaln2(false, 1, 1, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+iv*n-1).Matrix(n, opts), wr, zero, x); xnorm > one {
							if work.Get(j-1) > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							work.Off(1+iv*n-1).Scal(ki, scale, 1)
						}
						work.Set(j+iv*n-1, x.Get(0, 0))

						//                    Update right-hand side
						work.Off(1+iv*n-1).Axpy(j-1, -x.Get(0, 0), t.Off(0, j-1).Vector(), 1, 1)

					} else {
						//                    2-by-2 diagonal block
						if scale, xnorm, _ = Dlaln2(false, 2, 1, smin, one, t.Off(j-1-1, j-1-1), one, one, work.Off(j-1+iv*n-1).Matrix(n, opts), wr, zero, x); xnorm > one {
							beta = math.Max(work.Get(j-1-1), work.Get(j-1))
							if beta > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								x.Set(1, 0, x.Get(1, 0)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							work.Off(1+iv*n-1).Scal(ki, scale, 1)
						}
						work.Set(j-1+iv*n-1, x.Get(0, 0))
						work.Set(j+iv*n-1, x.Get(1, 0))

						//                    Update right-hand side
						work.Off(1+iv*n-1).Axpy(j-2, -x.Get(0, 0), t.Off(0, j-1-1).Vector(), 1, 1)
						work.Off(1+iv*n-1).Axpy(j-2, -x.Get(1, 0), t.Off(0, j-1).Vector(), 1, 1)
					}
				label60:
				}

				//              Copy the vector x or Q*x to VR and normalize.
				if !over {
					//                 ------------------------------
					//                 no back-transform: copy x to VR and normalize.
					vr.Off(0, is-1).Vector().Copy(ki, work.Off(1+iv*n-1), 1, 1)

					ii = vr.Off(0, is-1).Vector().Iamax(ki, 1)
					remax = one / math.Abs(vr.Get(ii-1, is-1))
					vr.Off(0, is-1).Vector().Scal(ki, remax, 1)

					for k = ki + 1; k <= n; k++ {
						vr.Set(k-1, is-1, zero)
					}

				} else if nb == 1 {
					//                 ------------------------------
					//                 version 1: back-transform each vector with GEMV, Q*x.
					if ki > 1 {
						if err = vr.Off(0, ki-1).Vector().Gemv(NoTrans, n, ki-1, one, vr, work.Off(1+iv*n-1), 1, work.Get(ki+iv*n-1), 1); err != nil {
							panic(err)
						}
					}

					ii = vr.Off(0, ki-1).Vector().Iamax(n, 1)
					remax = one / math.Abs(vr.Get(ii-1, ki-1))
					vr.Off(0, ki-1).Vector().Scal(n, remax, 1)

				} else {
					//                 ------------------------------
					//                 version 2: back-transform block of vectors with GEMM
					//                 zero out below vector
					for k = ki + 1; k <= n; k++ {
						work.Set(k+iv*n-1, zero)
					}
					iscomplex[iv-1] = ip
					//                 back-transform and normalization is done below
				}
			} else {
				//              --------------------------------------------------------
				//              Complex right eigenvector.
				//
				//              Initial solve
				//              [ ( T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I*WI) ]*X = 0.
				//              [ ( T(KI,  KI-1) T(KI,  KI) )               ]
				if math.Abs(t.Get(ki-1-1, ki-1)) >= math.Abs(t.Get(ki-1, ki-1-1)) {
					work.Set(ki-1+(iv-1)*n-1, one)
					work.Set(ki+iv*n-1, wi/t.Get(ki-1-1, ki-1))
				} else {
					work.Set(ki-1+(iv-1)*n-1, -wi/t.Get(ki-1, ki-1-1))
					work.Set(ki+iv*n-1, one)
				}
				work.Set(ki+(iv-1)*n-1, zero)
				work.Set(ki-1+iv*n-1, zero)

				//              Form right-hand side.
				for k = 1; k <= ki-2; k++ {
					work.Set(k+(iv-1)*n-1, -work.Get(ki-1+(iv-1)*n-1)*t.Get(k-1, ki-1-1))
					work.Set(k+iv*n-1, -work.Get(ki+iv*n-1)*t.Get(k-1, ki-1))
				}

				//              Solve upper quasi-triangular system:
				//              [ T(1:KI-2,1:KI-2) - (WR+i*WI) ]*X = SCALE*(WORK+i*WORK2)
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
						if scale, xnorm, _ = Dlaln2(false, 1, 2, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+(iv-1)*n-1).Matrix(n, opts), wr, wi, x); xnorm > one {
							if work.Get(j-1) > bignum/xnorm {
								x.Set(0, 0, x.Get(0, 0)/xnorm)
								x.Set(0, 1, x.Get(0, 1)/xnorm)
								scale = scale / xnorm
							}
						}

						//                    Scale if necessary
						if scale != one {
							work.Off(1+(iv-1)*n-1).Scal(ki, scale, 1)
							work.Off(1+iv*n-1).Scal(ki, scale, 1)
						}
						work.Set(j+(iv-1)*n-1, x.Get(0, 0))
						work.Set(j+iv*n-1, x.Get(0, 1))

						//                    Update the right-hand side
						work.Off(1+(iv-1)*n-1).Axpy(j-1, -x.Get(0, 0), t.Off(0, j-1).Vector(), 1, 1)
						work.Off(1+iv*n-1).Axpy(j-1, -x.Get(0, 1), t.Off(0, j-1).Vector(), 1, 1)

					} else {
						//                    2-by-2 diagonal block
						if scale, xnorm, _ = Dlaln2(false, 2, 2, smin, one, t.Off(j-1-1, j-1-1), one, one, work.Off(j-1+(iv-1)*n-1).Matrix(n, opts), wr, wi, x); xnorm > one {
							beta = math.Max(work.Get(j-1-1), work.Get(j-1))
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
							work.Off(1+(iv-1)*n-1).Scal(ki, scale, 1)
							work.Off(1+iv*n-1).Scal(ki, scale, 1)
						}
						work.Set(j-1+(iv-1)*n-1, x.Get(0, 0))
						work.Set(j+(iv-1)*n-1, x.Get(1, 0))
						work.Set(j-1+iv*n-1, x.Get(0, 1))
						work.Set(j+iv*n-1, x.Get(1, 1))

						//                    Update the right-hand side
						work.Off(1+(iv-1)*n-1).Axpy(j-2, -x.Get(0, 0), t.Off(0, j-1-1).Vector(), 1, 1)
						work.Off(1+(iv-1)*n-1).Axpy(j-2, -x.Get(1, 0), t.Off(0, j-1).Vector(), 1, 1)
						work.Off(1+iv*n-1).Axpy(j-2, -x.Get(0, 1), t.Off(0, j-1-1).Vector(), 1, 1)
						work.Off(1+iv*n-1).Axpy(j-2, -x.Get(1, 1), t.Off(0, j-1).Vector(), 1, 1)
					}
				label90:
				}

				//              Copy the vector x or Q*x to VR and normalize.
				if !over {
					//                 ------------------------------
					//                 no back-transform: copy x to VR and normalize.
					vr.Off(0, is-1-1).Vector().Copy(ki, work.Off(1+(iv-1)*n-1), 1, 1)
					vr.Off(0, is-1).Vector().Copy(ki, work.Off(1+iv*n-1), 1, 1)

					emax = zero
					for k = 1; k <= ki; k++ {
						emax = math.Max(emax, math.Abs(vr.Get(k-1, is-1-1))+math.Abs(vr.Get(k-1, is-1)))
					}
					remax = one / emax
					vr.Off(0, is-1-1).Vector().Scal(ki, remax, 1)
					vr.Off(0, is-1).Vector().Scal(ki, remax, 1)

					for k = ki + 1; k <= n; k++ {
						vr.Set(k-1, is-1-1, zero)
						vr.Set(k-1, is-1, zero)
					}

				} else if nb == 1 {
					//                 ------------------------------
					//                 version 1: back-transform each vector with GEMV, Q*x.
					if ki > 2 {
						if err = vr.Off(0, ki-1-1).Vector().Gemv(NoTrans, n, ki-2, one, vr, work.Off(1+(iv-1)*n-1), 1, work.Get(ki-1+(iv-1)*n-1), 1); err != nil {
							panic(err)
						}
						if err = vr.Off(0, ki-1).Vector().Gemv(NoTrans, n, ki-2, one, vr, work.Off(1+iv*n-1), 1, work.Get(ki+iv*n-1), 1); err != nil {
							panic(err)
						}
					} else {
						vr.Off(0, ki-1-1).Vector().Scal(n, work.Get(ki-1+(iv-1)*n-1), 1)
						vr.Off(0, ki-1).Vector().Scal(n, work.Get(ki+iv*n-1), 1)
					}

					emax = zero
					for k = 1; k <= n; k++ {
						emax = math.Max(emax, math.Abs(vr.Get(k-1, ki-1-1))+math.Abs(vr.Get(k-1, ki-1)))
					}
					remax = one / emax
					vr.Off(0, ki-1-1).Vector().Scal(n, remax, 1)
					vr.Off(0, ki-1).Vector().Scal(n, remax, 1)

				} else {
					//                 ------------------------------
					//                 version 2: back-transform block of vectors with GEMM
					//                 zero out below vector
					for k = ki + 1; k <= n; k++ {
						work.Set(k+(iv-1)*n-1, zero)
						work.Set(k+iv*n-1, zero)
					}
					iscomplex[iv-1-1] = -ip
					iscomplex[iv-1] = ip
					iv = iv - 1
					//                 back-transform and normalization is done below
				}
			}
			if nb > 1 {
				//              --------------------------------------------------------
				//              Blocked version of back-transform
				//              For complex case, KI2 includes both vectors (KI-1 and KI)
				if ip == 0 {
					ki2 = ki
				} else {
					ki2 = ki - 1
				}
				//              Columns IV:NB of work are valid vectors.
				//              When the number of vectors stored reaches NB-1 or NB,
				//              or if this was last vector, do the GEMM
				if (iv <= 2) || (ki2 == 1) {
					err = work.Off(1+(nb+iv)*n-1).Matrix(n, opts).Gemm(NoTrans, NoTrans, n, nb-iv+1, ki2+nb-iv, one, vr, work.Off(1+iv*n-1).Matrix(n, opts), zero)
					//                 normalize vectors
					for k = iv; k <= nb; k++ {
						if iscomplex[k-1] == 0 {
							//                       real eigenvector
							ii = work.Off(1+(nb+k)*n-1).Iamax(n, 1)
							remax = one / math.Abs(work.Get(ii+(nb+k)*n-1))
						} else if iscomplex[k-1] == 1 {
							//                       first eigenvector of conjugate pair
							emax = zero
							for ii = 1; ii <= n; ii++ {
								emax = math.Max(emax, math.Abs(work.Get(ii+(nb+k)*n-1))+math.Abs(work.Get(ii+(nb+k+1)*n-1)))
							}
							remax = one / emax
							//                    else if ISCOMPLEX(K).EQ.-1
							//                       second eigenvector of conjugate pair
							//                       reuse same REMAX as previous K
						}
						work.Off(1+(nb+k)*n-1).Scal(n, remax, 1)
					}
					Dlacpy(Full, n, nb-iv+1, work.Off(1+(nb+iv)*n-1).Matrix(n, opts), vr.Off(0, ki2-1))
					iv = nb
				} else {
					iv = iv - 1
				}
			}
			//! blocked back-transform

			is = is - 1
			if ip != 0 {
				is = is - 1
			}
		label140:
		}
	}
	if leftv {
		//        ============================================================
		//        Compute left eigenvectors.
		//
		//        IV is index of column in current block.
		//        For complex left vector, uses IV for real part and IV+1 for complex part.
		//        Non-blocked version always uses IV=1;
		//        blocked     version starts with IV=1, goes up to NB-1 or NB.
		//        (Note the "0-th" column is used for 1-norms computed above.)
		iv = 1
		ip = 0
		is = 1
		for ki = 1; ki <= n; ki++ {
			if ip == 1 {
				//              previous iteration (ki-1) was first of conjugate pair,
				//              so this ki is second of conjugate pair; skip to end of loop
				ip = -1
				goto label260
			} else if ki == n {
				//              last column, so this ki must be real eigenvalue
				ip = 0
			} else if t.Get(ki, ki-1) == zero {
				//              zero on sub-diagonal, so this ki is real eigenvalue
				ip = 0
			} else {
				//              non-zero on sub-diagonal, so this ki is first of conjugate pair
				ip = 1
			}

			if somev {
				if !(*_select)[ki-1] {
					goto label260
				}
			}

			//           Compute the KI-th eigenvalue (WR,WI).
			wr = t.Get(ki-1, ki-1)
			wi = zero
			if ip != 0 {
				wi = math.Sqrt(math.Abs(t.Get(ki-1, ki))) * math.Sqrt(math.Abs(t.Get(ki, ki-1)))
			}
			smin = math.Max(ulp*(math.Abs(wr)+math.Abs(wi)), smlnum)

			if ip == 0 {
				//              --------------------------------------------------------
				//              Real left eigenvector
				work.Set(ki+iv*n-1, one)

				//              Form right-hand side.
				for k = ki + 1; k <= n; k++ {
					work.Set(k+iv*n-1, -t.Get(ki-1, k-1))
				}

				//              Solve transposed quasi-triangular system:
				//              [ T(KI+1:N,KI+1:N) - WR ]**T * X = SCALE*WORK
				vmax = one
				vcrit = bignum

				jnxt = ki + 1
				for j = ki + 1; j <= n; j++ {
					if j < jnxt {
						goto label170
					}
					j1 = j
					j2 = j
					jnxt = j + 1
					if j < n {
						if t.Get(j, j-1) != zero {
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
							work.Off(ki+iv*n-1).Scal(n-ki+1, rec, 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+iv*n-1, work.Get(j+iv*n-1)-work.Off(ki+1+iv*n-1).Dot(j-ki-1, t.Off(ki, j-1).Vector(), 1, 1))

						//                    Solve [ T(J,J) - WR ]**T * X = WORK
						if scale, xnorm, _ = Dlaln2(false, 1, 1, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+iv*n-1).Matrix(n, opts), wr, zero, x); scale != one {
							work.Off(ki+iv*n-1).Scal(n-ki+1, scale, 1)
						}
						work.Set(j+iv*n-1, x.Get(0, 0))
						vmax = math.Max(math.Abs(work.Get(j+iv*n-1)), vmax)
						vcrit = bignum / vmax

					} else {
						//                    2-by-2 diagonal block
						//
						//                    Scale if necessary to avoid overflow when forming
						//                    the right-hand side.
						beta = math.Max(work.Get(j-1), work.Get(j))
						if beta > vcrit {
							rec = one / vmax
							work.Off(ki+iv*n-1).Scal(n-ki+1, rec, 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+iv*n-1, work.Get(j+iv*n-1)-work.Off(ki+1+iv*n-1).Dot(j-ki-1, t.Off(ki, j-1).Vector(), 1, 1))

						work.Set(j+1+iv*n-1, work.Get(j+1+iv*n-1)-work.Off(ki+1+iv*n-1).Dot(j-ki-1, t.Off(ki, j).Vector(), 1, 1))

						//                    Solve
						//                    [ T(J,J)-WR   T(J,J+1)      ]**T * X = SCALE*( WORK1 )
						//                    [ T(J+1,J)    T(J+1,J+1)-WR ]                ( WORK2 )
						if scale, xnorm, _ = Dlaln2(true, 2, 1, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+iv*n-1).Matrix(n, opts), wr, zero, x); scale != one {
							work.Off(ki+iv*n-1).Scal(n-ki+1, scale, 1)
						}
						work.Set(j+iv*n-1, x.Get(0, 0))
						work.Set(j+1+iv*n-1, x.Get(1, 0))

						vmax = math.Max(math.Abs(work.Get(j+iv*n-1)), math.Max(math.Abs(work.Get(j+1+iv*n-1)), vmax))
						vcrit = bignum / vmax

					}
				label170:
				}

				//              Copy the vector x or Q*x to VL and normalize.
				if !over {
					//                 ------------------------------
					//                 no back-transform: copy x to VL and normalize.
					vl.Off(ki-1, is-1).Vector().Copy(n-ki+1, work.Off(ki+iv*n-1), 1, 1)

					ii = vl.Off(ki-1, is-1).Vector().Iamax(n-ki+1, 1) + ki - 1
					remax = one / math.Abs(vl.Get(ii-1, is-1))
					vl.Off(ki-1, is-1).Vector().Scal(n-ki+1, remax, 1)

					for k = 1; k <= ki-1; k++ {
						t.Set(k-1, is-1, zero)
					}

				} else if nb == 1 {
					//                 ------------------------------
					//                 version 1: back-transform each vector with GEMV, Q*x.
					if ki < n {
						if err = vl.Off(0, ki-1).Vector().Gemv(NoTrans, n, n-ki, one, vl.Off(0, ki), work.Off(ki+1+iv*n-1), 1, work.Get(ki+iv*n-1), 1); err != nil {
							panic(err)
						}
					}

					ii = vl.Off(0, ki-1).Vector().Iamax(n, 1)
					remax = one / math.Abs(vl.Get(ii-1, ki-1))
					vl.Off(0, ki-1).Vector().Scal(n, remax, 1)

				} else {
					//                 ------------------------------
					//                 version 2: back-transform block of vectors with GEMM
					//                 zero out above vector
					//                 could go from KI-NV+1 to KI-1
					for k = 1; k <= ki-1; k++ {
						work.Set(k+iv*n-1, zero)
					}
					iscomplex[iv-1] = ip
					//                 back-transform and normalization is done below
				}
			} else {
				//              --------------------------------------------------------
				//              Complex left eigenvector.
				//
				//              Initial solve:
				//              [ ( T(KI,KI)    T(KI,KI+1)  )**T - (WR - I* WI) ]*X = 0.
				//              [ ( T(KI+1,KI) T(KI+1,KI+1) )                   ]
				if math.Abs(t.Get(ki-1, ki)) >= math.Abs(t.Get(ki, ki-1)) {
					work.Set(ki+iv*n-1, wi/t.Get(ki-1, ki))
					work.Set(ki+1+(iv+1)*n-1, one)
				} else {
					work.Set(ki+iv*n-1, one)
					work.Set(ki+1+(iv+1)*n-1, -wi/t.Get(ki, ki-1))
				}
				work.Set(ki+1+iv*n-1, zero)
				work.Set(ki+(iv+1)*n-1, zero)

				//              Form right-hand side.
				for k = ki + 2; k <= n; k++ {
					work.Set(k+iv*n-1, -work.Get(ki+iv*n-1)*t.Get(ki-1, k-1))
					work.Set(k+(iv+1)*n-1, -work.Get(ki+1+(iv+1)*n-1)*t.Get(ki, k-1))
				}

				//              Solve transposed quasi-triangular system:
				//              [ T(KI+2:N,KI+2:N)**T - (WR-i*WI) ]*X = WORK1+i*WORK2
				vmax = one
				vcrit = bignum

				jnxt = ki + 2
				for j = ki + 2; j <= n; j++ {
					if j < jnxt {
						goto label200
					}
					j1 = j
					j2 = j
					jnxt = j + 1
					if j < n {
						if t.Get(j, j-1) != zero {
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
							work.Off(ki+iv*n-1).Scal(n-ki+1, rec, 1)
							work.Off(ki+(iv+1)*n-1).Scal(n-ki+1, rec, 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+iv*n-1, work.Get(j+iv*n-1)-work.Off(ki+2+iv*n-1).Dot(j-ki-2, t.Off(ki+2-1, j-1).Vector(), 1, 1))
						work.Set(j+(iv+1)*n-1, work.Get(j+(iv+1)*n-1)-work.Off(ki+2+(iv+1)*n-1).Dot(j-ki-2, t.Off(ki+2-1, j-1).Vector(), 1, 1))

						//                    Solve [ T(J,J)-(WR-i*WI) ]*(X11+i*X12)= WK+I*WK2
						if scale, xnorm, _ = Dlaln2(false, 1, 2, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+iv*n-1).Matrix(n, opts), wr, -wi, x); scale != one {
							work.Off(ki+iv*n-1).Scal(n-ki+1, scale, 1)
							work.Off(ki+(iv+1)*n-1).Scal(n-ki+1, scale, 1)
						}
						work.Set(j+iv*n-1, x.Get(0, 0))
						work.Set(j+(iv+1)*n-1, x.Get(0, 1))
						vmax = math.Max(math.Abs(work.Get(j+iv*n-1)), math.Max(math.Abs(work.Get(j+(iv+1)*n-1)), vmax))
						vcrit = bignum / vmax

					} else {
						//                    2-by-2 diagonal block
						//
						//                    Scale if necessary to avoid overflow when forming
						//                    the right-hand side elements.
						beta = math.Max(work.Get(j-1), work.Get(j))
						if beta > vcrit {
							rec = one / vmax
							work.Off(ki+iv*n-1).Scal(n-ki+1, rec, 1)
							work.Off(ki+(iv+1)*n-1).Scal(n-ki+1, rec, 1)
							vmax = one
							vcrit = bignum
						}

						work.Set(j+iv*n-1, work.Get(j+iv*n-1)-work.Off(ki+2+iv*n-1).Dot(j-ki-2, t.Off(ki+2-1, j-1).Vector(), 1, 1))

						work.Set(j+(iv+1)*n-1, work.Get(j+(iv+1)*n-1)-work.Off(ki+2+(iv+1)*n-1).Dot(j-ki-2, t.Off(ki+2-1, j-1).Vector(), 1, 1))

						work.Set(j+1+iv*n-1, work.Get(j+1+iv*n-1)-work.Off(ki+2+iv*n-1).Dot(j-ki-2, t.Off(ki+2-1, j).Vector(), 1, 1))

						work.Set(j+1+(iv+1)*n-1, work.Get(j+1+(iv+1)*n-1)-work.Off(ki+2+(iv+1)*n-1).Dot(j-ki-2, t.Off(ki+2-1, j).Vector(), 1, 1))

						//                    Solve 2-by-2 complex linear equation
						//                    [ (T(j,j)   T(j,j+1)  )**T - (wr-i*wi)*I ]*X = SCALE*B
						//                    [ (T(j+1,j) T(j+1,j+1))                  ]
						if scale, xnorm, _ = Dlaln2(true, 2, 2, smin, one, t.Off(j-1, j-1), one, one, work.Off(j+iv*n-1).Matrix(n, opts), wr, -wi, x); scale != one {
							work.Off(ki+iv*n-1).Scal(n-ki+1, scale, 1)
							work.Off(ki+(iv+1)*n-1).Scal(n-ki+1, scale, 1)
						}
						work.Set(j+iv*n-1, x.Get(0, 0))
						work.Set(j+(iv+1)*n-1, x.Get(0, 1))
						work.Set(j+1+iv*n-1, x.Get(1, 0))
						work.Set(j+1+(iv+1)*n-1, x.Get(1, 1))
						vmax = math.Max(math.Abs(x.Get(0, 0)), math.Max(math.Abs(x.Get(0, 1)), math.Max(math.Abs(x.Get(1, 0)), math.Max(math.Abs(x.Get(1, 1)), vmax))))
						vcrit = bignum / vmax

					}
				label200:
				}

				//              Copy the vector x or Q*x to VL and normalize.
				if !over {
					//                 ------------------------------
					//                 no back-transform: copy x to VL and normalize.
					vl.Off(ki-1, is-1).Vector().Copy(n-ki+1, work.Off(ki+iv*n-1), 1, 1)
					vl.Off(ki-1, is).Vector().Copy(n-ki+1, work.Off(ki+(iv+1)*n-1), 1, 1)

					emax = zero
					for k = ki; k <= n; k++ {
						emax = math.Max(emax, math.Abs(vl.Get(k-1, is-1))+math.Abs(vl.Get(k-1, is)))
					}
					remax = one / emax
					vl.Off(ki-1, is-1).Vector().Scal(n-ki+1, remax, 1)
					vl.Off(ki-1, is).Vector().Scal(n-ki+1, remax, 1)

					for k = 1; k <= ki-1; k++ {
						t.Set(k-1, is-1, zero)
						t.Set(k-1, is, zero)
					}

				} else if nb == 1 {
					//                 ------------------------------
					//                 version 1: back-transform each vector with GEMV, Q*x.
					if ki < n-1 {
						if err = vl.Off(0, ki-1).Vector().Gemv(NoTrans, n, n-ki-1, one, vl.Off(0, ki+2-1), work.Off(ki+2+iv*n-1), 1, work.Get(ki+iv*n-1), 1); err != nil {
							panic(err)
						}
						if err = vl.Off(0, ki).Vector().Gemv(NoTrans, n, n-ki-1, one, vl.Off(0, ki+2-1), work.Off(ki+2+(iv+1)*n-1), 1, work.Get(ki+1+(iv+1)*n-1), 1); err != nil {
							panic(err)
						}
					} else {
						vl.Off(0, ki-1).Vector().Scal(n, work.Get(ki+iv*n-1), 1)
						vl.Off(0, ki).Vector().Scal(n, work.Get(ki+1+(iv+1)*n-1), 1)
					}

					emax = zero
					for k = 1; k <= n; k++ {
						emax = math.Max(emax, math.Abs(vl.Get(k-1, ki-1))+math.Abs(vl.Get(k-1, ki)))
					}
					remax = one / emax
					vl.Off(0, ki-1).Vector().Scal(n, remax, 1)
					vl.Off(0, ki).Vector().Scal(n, remax, 1)

				} else {
					//                 ------------------------------
					//                 version 2: back-transform block of vectors with GEMM
					//                 zero out above vector
					//                 could go from KI-NV+1 to KI-1
					for k = 1; k <= ki-1; k++ {
						work.Set(k+iv*n-1, zero)
						work.Set(k+(iv+1)*n-1, zero)
					}
					iscomplex[iv-1] = ip
					iscomplex[iv] = -ip
					iv = iv + 1
					//                 back-transform and normalization is done below
				}
			}
			if nb > 1 {
				//              --------------------------------------------------------
				//              Blocked version of back-transform
				//              For complex case, KI2 includes both vectors (KI and KI+1)
				if ip == 0 {
					ki2 = ki
				} else {
					ki2 = ki + 1
				}
				//              Columns 1:IV of work are valid vectors.
				//              When the number of vectors stored reaches NB-1 or NB,
				//              or if this was last vector, do the GEMM
				if (iv >= nb-1) || (ki2 == n) {
					if err = work.Off(1+(nb+1)*n-1).Matrix(n, opts).Gemm(NoTrans, NoTrans, n, iv, n-ki2+iv, one, vl.Off(0, ki2-iv), work.Off(ki2-iv+1+1*n-1).Matrix(n, opts), zero); err != nil {
						panic(err)
					}
					//                 normalize vectors
					for k = 1; k <= iv; k++ {
						if iscomplex[k-1] == 0 {
							//                       real eigenvector
							ii = work.Off(1+(nb+k)*n-1).Iamax(n, 1)
							remax = one / math.Abs(work.Get(ii+(nb+k)*n-1))
						} else if iscomplex[k-1] == 1 {
							//                       first eigenvector of conjugate pair
							emax = zero
							for ii = 1; ii <= n; ii++ {
								emax = math.Max(emax, math.Abs(work.Get(ii+(nb+k)*n-1))+math.Abs(work.Get(ii+(nb+k+1)*n-1)))
							}
							remax = one / emax
							//                    else if ISCOMPLEX(K).EQ.-1
							//                       second eigenvector of conjugate pair
							//                       reuse same REMAX as previous K
						}
						work.Off(1+(nb+k)*n-1).Scal(n, remax, 1)
					}
					Dlacpy(Full, n, iv, work.Off(1+(nb+1)*n-1).Matrix(n, opts), vl.Off(0, ki2-iv))
					iv = 1
				} else {
					iv = iv + 1
				}
			}
			//! blocked back-transform

			is = is + 1
			if ip != 0 {
				is = is + 1
			}
		label260:
		}
	}

	return
}
