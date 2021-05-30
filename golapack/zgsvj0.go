package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Zgsvj0 is called from ZGESVJ as a pre-processor and that is its main
// purpose. It applies Jacobi rotations in the same way as ZGESVJ does, but
// it does not check convergence (stopping criterion). Few tuning
// parameters (marked by [TP]) are available for the implementer.
func Zgsvj0(jobv byte, m, n *int, a *mat.CMatrix, lda *int, d *mat.CVector, sva *mat.Vector, mv *int, v *mat.CMatrix, ldv *int, eps, sfmin, tol *float64, nsweep *int, work *mat.CVector, lwork, info *int) {
	var applv, rotok, rsvec bool
	var aapq, ompq complex128
	var aapp, aapp0, aapq1, aaqq, apoaq, aqoap, big, bigtheta, cs, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, small, sn, t, temp1, theta, thsign, zero float64
	var blskip, emptsw, i, ibr, ierr, igl, ijblsk, ir1, iswrot, jbc, jgl, kbl, lkahead, mvl, nbl, notrot, p, pskipped, q, rowskip, swband int

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
		gltest.Xerbla([]byte("ZGSVJ0"), -(*info))
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

	//     .. Row-cyclic Jacobi SVD algorithm with column pivoting ..
	emptsw = ((*n) * ((*n) - 1)) / 2
	notrot = 0

	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	swband = 0
	//[TP] SWBAND is a tuning parameter [TP]. It is meaningful and effective
	//     if ZGESVJ is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm ZGEJSV. For sweeps i=1:SWBAND the procedure
	//     works on pivots inside a band-like region around the diagonal.
	//     The boundaries are determined dynamically, based on the number of
	//     pivots above a threshold.

	kbl = minint(int(8), *n)
	//[TP] KBL is a tuning parameter that defines the tile size in the
	//     tiling of the p-q loops of pivot pairs. In general, an optimal
	//     value of KBL depends on the matrix dimensions and on the
	//     parameters of the computer's memory.

	nbl = (*n) / kbl
	if (nbl * kbl) != (*n) {
		nbl = nbl + 1
	}

	blskip = powint(kbl, 2)
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.

	rowskip = minint(5, kbl)
	//[TP] ROWSKIP is a tuning parameter.

	lkahead = 1
	//[TP] LKAHEAD is a tuning parameter.
	//
	//     Quasi block transformations, using the lower (upper) triangular
	//     structure of the input matrix. The quasi-block-cycling usually
	//     invokes cubic convergence. Big part of this cycle is done inside
	//     canonical subspaces of dimensions less than M.
	//
	//
	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	for i = 1; i <= (*nsweep); i++ {
		//     .. go go go ...
		mxaapq = zero
		mxsinj = zero
		iswrot = 0

		notrot = 0
		pskipped = 0

		//     Each sweep is unrolled using KBL-by-KBL tiles over the pivot pairs
		//     1 <= p < q <= N. This is the first step toward a blocked implementation
		//     of the rotations. New implementation, based on block transformations,
		//     is under development.
		for ibr = 1; ibr <= nbl; ibr++ {

			igl = (ibr-1)*kbl + 1

			for ir1 = 0; ir1 <= minint(lkahead, nbl-ibr); ir1++ {

				igl = igl + ir1*kbl

				for p = igl; p <= minint(igl+kbl-1, (*n)-1); p++ {
					//     .. de Rijk's pivoting
					q = goblas.Idamax(toPtr((*n)-p+1), sva.Off(p-1), func() *int { y := 1; return &y }()) + p - 1
					if p != q {
						goblas.Zswap(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }())
						if rsvec {
							goblas.Zswap(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }())
						}
						temp1 = sva.Get(p - 1)
						sva.Set(p-1, sva.Get(q-1))
						sva.Set(q-1, temp1)
						aapq = d.Get(p - 1)
						d.Set(p-1, d.Get(q-1))
						d.Set(q-1, aapq)
					}

					if ir1 == 0 {
						//        Column norms are periodically updated by explicit
						//        norm computation.
						//        Caveat:
						//        Unfortunately, some BLAS implementations compute SNCRM2(M,A(1,p),1)
						//        as SQRT(S=ZDOTC(M,A(1,p),1,A(1,p),1)), which may cause the result to
						//        overflow for ||A(:,p)||_2 > SQRT(overflow_threshold), and to
						//        underflow for ||A(:,p)||_2 < SQRT(underflow_threshold).
						//        Hence, DZNRM2 cannot be trusted, not even in the case when
						//        the true norm is far from the under(over)flow boundaries.
						//        If properly implemented DZNRM2 is available, the IF-THEN-ELSE-END IF
						//        below should be replaced with "AAPP = DZNRM2( M, A(1,p), 1 )".
						if (sva.Get(p-1) < rootbig) && (sva.Get(p-1) > rootsfmin) {
							sva.Set(p-1, goblas.Dznrm2(m, a.CVector(0, p-1), func() *int { y := 1; return &y }()))
						} else {
							temp1 = zero
							aapp = one
							Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &temp1, &aapp)
							sva.Set(p-1, temp1*math.Sqrt(aapp))
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
										aapq = (goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(aaqq, 0)) / complex(aapp, 0)
									} else {
										goblas.Zcopy(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(m, work, func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(aaqq, 0)
									}
								} else {
									rotok = aapp <= (aaqq / small)
									if aapp > (small / aaqq) {
										aapq = (goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(aapp, 0)) / complex(aaqq, 0)
									} else {
										goblas.Zcopy(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }()) / complex(aapp, 0)
									}
								}

								//                           AAPQ = AAPQ * CONJG( CWORK(p) ) * CWORK(q)
								aapq1 = -cmplx.Abs(aapq)
								mxaapq = maxf64(mxaapq, -aapq1)

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq1) > (*tol) {
									ompq = aapq / complex(cmplx.Abs(aapq), 0)

									//           .. rotate
									//[RTD]      ROTATED = ROTATED + ONE
									if ir1 == 0 {
										notrot = 0
										pskipped = 0
										iswrot = iswrot + 1
									}

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq1

										if math.Abs(theta) > bigtheta {

											t = half / theta
											cs = one
											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))
											mxsinj = maxf64(mxsinj, math.Abs(t))

										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq1)
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs

											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))

											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											}
										}
										d.Set(p-1, -d.Get(q-1)*ompq)

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										goblas.Zcopy(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
										goblas.Zaxpy(m, toPtrc128(-aapq), work, func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aaqq, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
										sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
										mxsinj = maxf64(mxsinj, *sfmin)
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           recompute SVA(q), SVA(p).

									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dznrm2(m, a.CVector(0, q-1), func() *int { y := 1; return &y }()))
										} else {
											t = zero
											aaqq = one
											Zlassq(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq))
										}
									}
									if (aapp / aapp0) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dznrm2(m, a.CVector(0, p-1), func() *int { y := 1; return &y }())
										} else {
											t = zero
											aapp = one
											Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &t, &aapp)
											aapp = t * math.Sqrt(aapp)
										}
										sva.Set(p-1, aapp)
									}

								} else {
									//        A(:,p) and A(:,q) already numerically orthogonal
									if ir1 == 0 {
										notrot = notrot + 1
									}
									//[RTD]      SKIPPED  = SKIPPED  + 1
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

								//     .. M x 2 Jacobi SVD ..
								//
								//        Safe Gram matrix computation
								if aaqq >= one {
									if aapp >= aaqq {
										rotok = (small * aapp) <= aaqq
									} else {
										rotok = (small * aaqq) <= aapp
									}
									if aapp < (big / aaqq) {
										aapq = (goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(aaqq, 0)) / complex(aapp, 0)
									} else {
										goblas.Zcopy(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(m, work, func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(aaqq, 0)
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }()) / complex(maxf64(aaqq, aapp), 0)) / complex(minf64(aaqq, aapp), 0)
									} else {
										goblas.Zcopy(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }()) / complex(aapp, 0)
									}
								}

								//                           AAPQ = AAPQ * CONJG(CWORK(p))*CWORK(q)
								aapq1 = -cmplx.Abs(aapq)
								mxaapq = maxf64(mxaapq, -aapq1)

								//        TO rotate or NOT to rotate, THAT is the question ...
								if math.Abs(aapq1) > (*tol) {
									ompq = aapq / complex(cmplx.Abs(aapq), 0)
									notrot = 0
									//[RTD]      ROTATED  = ROTATED + 1
									pskipped = 0
									iswrot = iswrot + 1

									if rotok {

										aqoap = aaqq / aapp
										apoaq = aapp / aaqq
										theta = -half * math.Abs(aqoap-apoaq) / aapq1
										if aaqq > aapp0 {
											theta = -theta
										}

										if math.Abs(theta) > bigtheta {
											t = half / theta
											cs = one
											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(t, 0)))
											}
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))
											mxsinj = maxf64(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq1)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = maxf64(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(maxf64(zero, one-t*aqoap*aapq1))

											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											}
										}
										d.Set(p-1, -d.Get(q-1)*ompq)

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										if aapp > aaqq {
											goblas.Zcopy(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											goblas.Zaxpy(m, toPtrc128(-aapq), work, func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }())
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aaqq, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											sva.Set(q-1, aaqq*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
											mxsinj = maxf64(mxsinj, *sfmin)
										} else {
											goblas.Zcopy(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), work, func() *int { y := 1; return &y }())
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											goblas.Zaxpy(m, toPtrc128(-cmplx.Conj(aapq)), work, func() *int { y := 1; return &y }(), a.CVector(0, p-1), func() *int { y := 1; return &y }())
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aapp, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											sva.Set(p-1, aapp*math.Sqrt(maxf64(zero, one-aapq1*aapq1)))
											mxsinj = maxf64(mxsinj, *sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           .. recompute SVA(q), SVA(p)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dznrm2(m, a.CVector(0, q-1), func() *int { y := 1; return &y }()))
										} else {
											t = zero
											aaqq = one
											Zlassq(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dznrm2(m, a.CVector(0, p-1), func() *int { y := 1; return &y }())
										} else {
											t = zero
											aapp = one
											Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &t, &aapp)
											aapp = t * math.Sqrt(aapp)
										}
										sva.Set(p-1, aapp)
									}
									//              end of OK rotation
								} else {
									notrot = notrot + 1
									//[RTD]      SKIPPED  = SKIPPED  + 1
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
				sva.Set(p-1, sva.GetMag(p-1))
			}

		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get((*n)-1) < rootbig) && (sva.Get((*n)-1) > rootsfmin) {
			sva.Set((*n)-1, goblas.Dznrm2(m, a.CVector(0, (*n)-1), func() *int { y := 1; return &y }()))
		} else {
			t = zero
			aapp = one
			Zlassq(m, a.CVector(0, (*n)-1), func() *int { y := 1; return &y }(), &t, &aapp)
			sva.Set((*n)-1, t*math.Sqrt(aapp))
		}

		//     Additional steering devices
		if (i < swband) && ((mxaapq <= roottol) || (iswrot <= (*n))) {
			swband = i
		}

		if (i > swband+1) && (mxaapq < math.Sqrt(float64(*n))*(*tol)) && (float64(*n)*mxaapq*mxsinj < (*tol)) {
			goto label1994
		}

		if notrot >= emptsw {
			goto label1994
		}

	}
	//     end i=1:NSWEEP loop
	//
	// #:( Reaching this point means that the procedure has not converged.
	(*info) = (*nsweep) - 1
	goto label1995

label1994:
	;
	// #:) Reaching this point means numerical convergence after the i-th
	//     sweep.
	(*info) = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the vector SVA() of column norms.
	for p = 1; p <= (*n)-1; p++ {
		q = goblas.Idamax(toPtr((*n)-p+1), sva.Off(p-1), func() *int { y := 1; return &y }()) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			aapq = d.Get(p - 1)
			d.Set(p-1, d.Get(q-1))
			d.Set(q-1, aapq)
			goblas.Zswap(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }())
			if rsvec {
				goblas.Zswap(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }())
			}
		}
	}
}
