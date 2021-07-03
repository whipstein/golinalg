package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgsvj0 is called from DGESVJ as a pre-processor and that is its main
// purpose. It applies Jacobi rotations in the same way as DGESVJ does, but
// it does not check convergence (stopping criterion). Few tuning
// parameters (marked by [TP]) are available for the implementer.
func Dgsvj0(jobv byte, m, n *int, a *mat.Matrix, lda *int, d, sva *mat.Vector, mv *int, v *mat.Matrix, ldv *int, eps, sfmin, tol *float64, nsweep *int, work *mat.Vector, lwork, info *int) {
	var applv, rotok, rsvec bool
	var aapp, aapp0, aapq, aaqq, apoaq, aqoap, big, bigtheta, cs, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, small, sn, t, temp1, theta, thsign, zero float64
	var blskip, emptsw, i, ibr, ierr, igl, ijblsk, ir1, iswrot, jbc, jgl, kbl, lkahead, mvl, nbl, notrot, p, pskipped, q, rowskip, swband int

	fastr := mat.NewDrotMatrix()

	zero = 0.0
	half = 0.5
	one = 1.0

	//     Test the input parameters.
	applv = jobv == 'A'
	rsvec = jobv == 'V'
	if !(rsvec || applv || jobv == 'N') {
		(*info) = -1
	} else if (*m) < 0 {
		(*info) = -2
	} else if ((*n) < 0) || ((*n) > (*m)) {
		(*info) = -3
	} else if (*lda) < (*m) {
		(*info) = -5
	} else if (rsvec || applv) && ((*mv) < 0) {
		(*info) = -8
	} else if (rsvec && ((*ldv) < (*n))) || (applv && ((*ldv) < (*mv))) {
		(*info) = -10
	} else if (*tol) <= (*eps) {
		(*info) = -13
	} else if (*nsweep) < 0 {
		(*info) = -14
	} else if (*lwork) < (*m) {
		(*info) = -16
	} else {
		(*info) = 0
	}

	//     #:(
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGSVJ0"), -(*info))
		return
	}

	if rsvec {
		mvl = (*n)
	} else if applv {
		mvl = (*mv)
	}
	rsvec = rsvec || applv
	rooteps = math.Sqrt(*eps)
	rootsfmin = math.Sqrt(*sfmin)
	small = (*sfmin) / (*eps)
	big = one / (*sfmin)
	rootbig = one / rootsfmin
	bigtheta = one / rooteps
	roottol = math.Sqrt(*tol)

	//     -#- Row-cyclic Jacobi SVD algorithm with column pivoting -#-
	emptsw = ((*n) * ((*n) - 1)) / 2
	notrot = 0
	fastr.Flag = int(zero)

	//     -#- Row-cyclic pivot strategy with de Rijk's pivoting -#-
	swband = 0
	//[TP] SWBAND is a tuning parameter. It is meaningful and effective
	//     if SGESVJ is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm SGESVJ. For sweeps i=1:SWBAND the procedure
	//     ......
	kbl = minint(int(8), *n)
	//[TP] KBL is a tuning parameter that defines the tile size in the
	//     tiling of the p-q loops of pivot pairs. In general, an optimal
	//     value of KBL depends on the matrix dimensions and on the
	//     parameters of the computer's memory.
	//
	nbl = (*n) / kbl
	if (nbl * kbl) != (*n) {
		nbl = nbl + 1
	}
	blskip = int(math.Pow(float64(kbl), 2) + 1)
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.
	rowskip = minint(int(5), kbl)
	//[TP] ROWSKIP is a tuning parameter.
	lkahead = 1
	//[TP] LKAHEAD is a tuning parameter.
	swband = 0
	pskipped = 0

	for i = 1; i <= (*nsweep); i++ {
		//     .. go go go ...

		mxaapq = zero
		mxsinj = zero
		iswrot = 0

		notrot = 0
		pskipped = 0

		for ibr = 1; ibr <= nbl; ibr++ {
			igl = (ibr-1)*kbl + 1

			for ir1 = 0; ir1 <= minint(lkahead, nbl-ibr); ir1++ {

				igl = igl + ir1*kbl

				for p = igl; p <= minint(igl+kbl-1, (*n)-1); p++ {
					//     .. de Rijk's pivoting
					q = goblas.Idamax((*n)-p+1, sva.Off(p-1), 1) + p - 1
					if p != q {
						goblas.Dswap(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
						if rsvec {
							goblas.Dswap(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
						}
						temp1 = sva.Get(p - 1)
						sva.Set(p-1, sva.Get(q-1))
						sva.Set(q-1, temp1)
						temp1 = d.Get(p - 1)
						d.Set(p-1, d.Get(q-1))
						d.Set(q-1, temp1)
					}

					if ir1 == 0 {
						//        Column norms are periodically updated by explicit
						//        norm computation.
						//        Caveat:
						//        Some BLAS implementations compute DNRM2(M,A(1,p),1)
						//        as DSQRT(DDOT(M,A(1,p),1,A(1,p),1)), which may result in
						//        overflow for ||A(:,p)||_2 > DSQRT(overflow_threshold), and
						//        undeflow for ||A(:,p)||_2 < DSQRT(underflow_threshold).
						//        Hence, DNRM2 cannot be trusted, not even in the case when
						//        the true norm is far from the under(over)flow boundaries.
						//        If properly implemented DNRM2 is available, the IF-THEN-ELSE
						//        below should read "AAPP = DNRM2( M, A(1,p), 1 ) * D(p)".
						if (sva.Get(p-1) < rootbig) && (sva.Get(p-1) > rootsfmin) {
							sva.Set(p-1, goblas.Dnrm2(*m, a.Vector(0, p-1), 1)*d.Get(p-1))
						} else {
							temp1 = zero
							aapp = one
							Dlassq(m, a.Vector(0, p-1), toPtr(1), &temp1, &aapp)
							sva.Set(p-1, temp1*math.Sqrt(aapp)*d.Get(p-1))
						}
						aapp = sva.Get(p - 1)
					} else {
						aapp = sva.Get(p - 1)
					}

					if aapp > zero {

						pskipped = 0

						for q = p + 1; q <= minint(igl+kbl-1, *n); q++ {

							aaqq = sva.Get(q - 1)
							if aaqq > zero {

								aapp0 = aapp
								if aaqq >= one {
									rotok = (small * aapp) <= aaqq
									if aapp < (big / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, p-1), 1, work, 1)
										Dlascl('G', toPtr(0), toPtr(0), &aapp, d.GetPtr(p-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, 1, a.Vector(0, q-1), 1) * d.Get(q-1) / aaqq
									}
								} else {
									rotok = aapp <= (aaqq / small)
									if aapp > (small / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, q-1), 1, work, 1)
										Dlascl('G', toPtr(0), toPtr(0), &aaqq, d.GetPtr(q-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, 1, a.Vector(0, p-1), 1) * d.Get(p-1) / aapp
									}
								}

								mxaapq = maxf64(mxaapq, math.Abs(aapq))

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq) > (*tol) {
									//           .. rotate
									//           ROTATED = ROTATED + ONE
									if ir1 == 0 {
										notrot = 0
										pskipped = 0
										iswrot = iswrot + 1
									}

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq

										if math.Abs(theta) > bigtheta {

											t = half / theta
											fastr.H21 = t * d.Get(p-1) / d.Get(q-1)
											fastr.H12 = -t * d.Get(q-1) / d.Get(p-1)
											goblas.Drotm(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1, fastr)
											if rsvec {
												goblas.Drotm(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1, fastr)
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq))
											mxsinj = maxf64(mxsinj, math.Abs(t))

										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -signf64(one, aapq)
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs

											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq))

											apoaq = d.Get(p-1) / d.Get(q-1)
											aqoap = d.Get(q-1) / d.Get(p-1)
											if d.Get(p-1) >= one {
												if d.Get(q-1) >= one {
													fastr.H21 = t * apoaq
													fastr.H12 = -t * aqoap
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)*cs)
													goblas.Drotm(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1, fastr)
													if rsvec {
														goblas.Drotm(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1, fastr)
													}
												} else {
													goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
													goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)/cs)
													if rsvec {
														goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
														goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
													}
												}
											} else {
												if d.Get(q-1) >= one {
													goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
													goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
													d.Set(p-1, d.Get(p-1)/cs)
													d.Set(q-1, d.Get(q-1)*cs)
													if rsvec {
														goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
														goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
													}
												} else {
													if d.Get(p-1) >= d.Get(q-1) {
														goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
														goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
														d.Set(p-1, d.Get(p-1)*cs)
														d.Set(q-1, d.Get(q-1)/cs)
														if rsvec {
															goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
															goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
														}
													} else {
														goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
														goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
														d.Set(p-1, d.Get(p-1)/cs)
														d.Set(q-1, d.Get(q-1)*cs)
														if rsvec {
															goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
															goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
														}
													}
												}
											}
										}

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										goblas.Dcopy(*m, a.Vector(0, p-1), 1, work, 1)
										Dlascl('G', toPtr(0), toPtr(0), &aapp, &one, m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										Dlascl('G', toPtr(0), toPtr(0), &aaqq, &one, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
										temp1 = -aapq * d.Get(p-1) / d.Get(q-1)
										goblas.Daxpy(*m, temp1, work, 1, a.Vector(0, q-1), 1)
										Dlascl('G', toPtr(0), toPtr(0), &one, &aaqq, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
										sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq*aapq)))
										mxsinj = maxf64(mxsinj, *sfmin)
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           recompute SVA(q), SVA(p).
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dnrm2(*m, a.Vector(0, q-1), 1)*d.Get(q-1))
										} else {
											t = zero
											aaqq = one
											Dlassq(m, a.Vector(0, q-1), toPtr(1), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*d.Get(q-1))
										}
									}
									if (aapp / aapp0) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dnrm2(*m, a.Vector(0, p-1), 1) * d.Get(p-1)
										} else {
											t = zero
											aapp = one
											Dlassq(m, a.Vector(0, p-1), toPtr(1), &t, &aapp)
											aapp = t * math.Sqrt(aapp) * d.Get(p-1)
										}
										sva.Set(p-1, aapp)
									}

								} else {
									//        A(:,p) and A(:,q) already numerically orthogonal
									if ir1 == 0 {
										notrot = notrot + 1
									}
									pskipped = pskipped + 1
								}
							} else {
								//        A(:,q) is zero column
								if ir1 == 0 {
									notrot = notrot + 1
								}
								pskipped = pskipped + 1
							}

							if (i <= swband) && (pskipped > rowskip) {
								if ir1 == 0 {
									aapp = -aapp
								}
								notrot = 0
								goto label2103
							}

						}
						//     END q-LOOP

					label2103:
						;
						//     bailed out of q-loop
						sva.Set(p-1, aapp)
					} else {
						sva.Set(p-1, aapp)
						if (ir1 == 0) && (aapp == zero) {
							notrot = notrot + minint(igl+kbl-1, *n) - p
						}
					}

				}
				//     end of the p-loop
				//     end of doing the block ( ibr, ibr )
			}
			//     end of ir1-loop
			//
			//........................................................
			// ... go to the off diagonal blocks

			igl = (ibr-1)*kbl + 1

			for jbc = ibr + 1; jbc <= nbl; jbc++ {

				jgl = (jbc-1)*kbl + 1

				//        doing the block at ( ibr, jbc )
				ijblsk = 0
				for p = igl; p <= minint(igl+kbl-1, *n); p++ {

					aapp = sva.Get(p - 1)

					if aapp > zero {

						pskipped = 0

						for q = jgl; q <= minint(jgl+kbl-1, *n); q++ {

							aaqq = sva.Get(q - 1)

							if aaqq > zero {
								aapp0 = aapp

								//     -#- M x 2 Jacobi SVD -#-
								//
								//        -#- Safe Gram matrix computation -#-
								if aaqq >= one {
									if aapp >= aaqq {
										rotok = (small * aapp) <= aaqq
									} else {
										rotok = (small * aaqq) <= aapp
									}
									if aapp < (big / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, p-1), 1, work, 1)
										Dlascl('G', toPtr(0), toPtr(0), &aapp, d.GetPtr(p-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, 1, a.Vector(0, q-1), 1) * d.Get(q-1) / aaqq
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, q-1), 1, work, 1)
										Dlascl('G', toPtr(0), toPtr(0), &aaqq, d.GetPtr(q-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, 1, a.Vector(0, p-1), 1) * d.Get(p-1) / aapp
									}
								}

								mxaapq = maxf64(mxaapq, math.Abs(aapq))

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq) > (*tol) {
									notrot = 0
									//           ROTATED  = ROTATED + 1
									pskipped = 0
									iswrot = iswrot + 1

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq
										if aaqq > aapp0 {
											theta = -theta
										}

										if math.Abs(theta) > bigtheta {
											t = half / theta
											fastr.H21 = t * d.Get(p-1) / d.Get(q-1)
											fastr.H12 = -t * d.Get(q-1) / d.Get(p-1)
											goblas.Drotm(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1, fastr)
											if rsvec {
												goblas.Drotm(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1, fastr)
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq))
											mxsinj = maxf64(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -signf64(one, aapq)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq))

											apoaq = d.Get(p-1) / d.Get(q-1)
											aqoap = d.Get(q-1) / d.Get(p-1)
											if d.Get(p-1) >= one {

												if d.Get(q-1) >= one {
													fastr.H21 = t * apoaq
													fastr.H12 = -t * aqoap
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)*cs)
													goblas.Drotm(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1, fastr)
													if rsvec {
														goblas.Drotm(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1, fastr)
													}
												} else {
													goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
													goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
													if rsvec {
														goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
														goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
													}
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)/cs)
												}
											} else {
												if d.Get(q-1) >= one {
													goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
													goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
													if rsvec {
														goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
														goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
													}
													d.Set(p-1, d.Get(p-1)/cs)
													d.Set(q-1, d.Get(q-1)*cs)
												} else {
													if d.Get(p-1) >= d.Get(q-1) {
														goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
														goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
														d.Set(p-1, d.Get(p-1)*cs)
														d.Set(q-1, d.Get(q-1)/cs)
														if rsvec {
															goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
															goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
														}
													} else {
														goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
														goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1), 1, a.Vector(0, p-1), 1)
														d.Set(p-1, d.Get(p-1)/cs)
														d.Set(q-1, d.Get(q-1)*cs)
														if rsvec {
															goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
															goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1), 1, v.Vector(0, p-1), 1)
														}
													}
												}
											}
										}

									} else {
										if aapp > aaqq {
											goblas.Dcopy(*m, a.Vector(0, p-1), 1, work, 1)
											Dlascl('G', toPtr(0), toPtr(0), &aapp, &one, m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
											Dlascl('G', toPtr(0), toPtr(0), &aaqq, &one, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
											temp1 = -aapq * d.Get(p-1) / d.Get(q-1)
											goblas.Daxpy(*m, temp1, work, 1, a.Vector(0, q-1), 1)
											Dlascl('G', toPtr(0), toPtr(0), &one, &aaqq, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq*aapq)))
											mxsinj = maxf64(mxsinj, *sfmin)
										} else {
											goblas.Dcopy(*m, a.Vector(0, q-1), 1, work, 1)
											Dlascl('G', toPtr(0), toPtr(0), &aaqq, &one, m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
											Dlascl('G', toPtr(0), toPtr(0), &aapp, &one, m, toPtr(1), a.Off(0, p-1), lda, &ierr)
											temp1 = -aapq * d.Get(q-1) / d.Get(p-1)
											goblas.Daxpy(*m, temp1, work, 1, a.Vector(0, p-1), 1)
											Dlascl('G', toPtr(0), toPtr(0), &one, &aapp, m, toPtr(1), a.Off(0, p-1), lda, &ierr)
											sva.Set(p-1, aapp*math.Sqrt(maxf64(zero, one-aapq*aapq)))
											mxsinj = maxf64(mxsinj, *sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q)
									//           .. recompute SVA(q)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dnrm2(*m, a.Vector(0, q-1), 1)*d.Get(q-1))
										} else {
											t = zero
											aaqq = one
											Dlassq(m, a.Vector(0, q-1), toPtr(1), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*d.Get(q-1))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dnrm2(*m, a.Vector(0, p-1), 1) * d.Get(p-1)
										} else {
											t = zero
											aapp = one
											Dlassq(m, a.Vector(0, p-1), toPtr(1), &t, &aapp)
											aapp = t * math.Sqrt(aapp) * d.Get(p-1)
										}
										sva.Set(p-1, aapp)
									}
									//              end of OK rotation
								} else {
									notrot = notrot + 1
									pskipped = pskipped + 1
									ijblsk = ijblsk + 1
								}
							} else {
								notrot = notrot + 1
								pskipped = pskipped + 1
								ijblsk = ijblsk + 1
							}

							if (i <= swband) && (ijblsk >= blskip) {
								sva.Set(p-1, aapp)
								notrot = 0
								goto label2011
							}
							if (i <= swband) && (pskipped > rowskip) {
								aapp = -aapp
								notrot = 0
								goto label2203
							}

						}
						//        end of the q-loop
					label2203:
						;

						sva.Set(p-1, aapp)

					} else {
						if aapp == zero {
							notrot = notrot + minint(jgl+kbl-1, *n) - jgl + 1
						}
						if aapp < zero {
							notrot = 0
						}
					}
				}
				//     end of the p-loop
			}
			//     end of the jbc-loop
		label2011:
			;
			//2011 bailed out of the jbc-loop
			for p = igl; p <= minint(igl+kbl-1, *n); p++ {
				sva.Set(p-1, math.Abs(sva.Get(p-1)))
			}

		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get((*n)-1) < rootbig) && (sva.Get((*n)-1) > rootsfmin) {
			sva.Set((*n)-1, goblas.Dnrm2(*m, a.Vector(0, (*n)-1), 1)*d.Get((*n)-1))
		} else {
			t = zero
			aapp = one
			Dlassq(m, a.Vector(0, (*n)-1), toPtr(1), &t, &aapp)
			sva.Set((*n)-1, t*math.Sqrt(aapp)*d.Get((*n)-1))
		}

		//     Additional steering devices
		if (i < swband) && ((mxaapq <= roottol) || (iswrot <= (*n))) {
			swband = i
		}

		if (i > swband+1) && (mxaapq < float64(*n)*(*tol)) && (float64(*n)*mxaapq*mxsinj < (*tol)) {
			goto label1994
		}

		if notrot >= emptsw {
			goto label1994
		}
	}
	//     end i=1:NSWEEP loop
	// #:) Reaching this point means that the procedure has completed the given
	//     number of iterations.
	(*info) = (*nsweep) - 1
	goto label1995
label1994:
	;
	// #:) Reaching this point means that during the i-th sweep all pivots were
	//     below the given tolerance, causing early exit.

	(*info) = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the vector D.
	for p = 1; p <= (*n)-1; p++ {
		q = goblas.Idamax((*n)-p+1, sva.Off(p-1), 1) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			temp1 = d.Get(p - 1)
			d.Set(p-1, d.Get(q-1))
			d.Set(q-1, temp1)
			goblas.Dswap(*m, a.Vector(0, p-1), 1, a.Vector(0, q-1), 1)
			if rsvec {
				goblas.Dswap(mvl, v.Vector(0, p-1), 1, v.Vector(0, q-1), 1)
			}
		}
	}
}
