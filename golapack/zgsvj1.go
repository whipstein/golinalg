package golapack

import (
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zgsvj1 is called from ZGESVJ as a pre-processor and that is its main
// purpose. It applies Jacobi rotations in the same way as ZGESVJ does, but
// it targets only particular pivots and it does not check convergence
// (stopping criterion). Few tunning parameters (marked by [TP]) are
// available for the implementer.
//
// Further Details
// ~~~~~~~~~~~~~~~
// ZGSVJ1 applies few sweeps of Jacobi rotations in the column space of
// the input M-by-N matrix A. The pivot pairs are taken from the (1,2)
// off-diagonal block in the corresponding N-by-N Gram matrix A^T * A. The
// block-entries (tiles) of the (1,2) off-diagonal block are marked by the
// [x]'s in the following scheme:
//
//    | *  *  * [x] [x] [x]|
//    | *  *  * [x] [x] [x]|    Row-cycling in the nblr-by-nblc [x] blocks.
//    | *  *  * [x] [x] [x]|    Row-cyclic pivoting inside each [x] block.
//    |[x] [x] [x] *  *  * |
//    |[x] [x] [x] *  *  * |
//    |[x] [x] [x] *  *  * |
//
// In terms of the columns of A, the first N1 columns are rotated 'against'
// the remaining N-N1 columns, trying to increase the angle between the
// corresponding subspaces. The off-diagonal block is N1-by(N-N1) and it is
// tiled using quadratic tiles of side KBL. Here, KBL is a tunning parameter.
// The number of sweeps is given in NSWEEP and the orthogonality threshold
// is given in TOL.
func Zgsvj1(jobv byte, m, n, n1 *int, a *mat.CMatrix, lda *int, d *mat.CVector, sva *mat.Vector, mv *int, v *mat.CMatrix, ldv *int, eps, sfmin, tol *float64, nsweep *int, work *mat.CVector, lwork, info *int) {
	var applv, rotok, rsvec bool
	var aapq, ompq complex128
	var aapp, aapp0, aapq1, aaqq, apoaq, aqoap, big, bigtheta, cs, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, small, sn, t, temp1, theta, thsign, zero float64
	var blskip, emptsw, i, ibr, ierr, igl, ijblsk, iswrot, jbc, jgl, kbl, mvl, nblc, nblr, notrot, p, pskipped, q, rowskip, swband int

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
	} else if (*n1) < 0 {
		(*info) = -4
	} else if (*lda) < (*m) {
		(*info) = -6
	} else if (rsvec || applv) && ((*mv) < 0) {
		(*info) = -9
	} else if (rsvec && ((*ldv) < (*n))) || (applv && ((*ldv) < (*mv))) {
		(*info) = -11
	} else if (*tol) <= (*eps) {
		(*info) = -14
	} else if (*nsweep) < 0 {
		(*info) = -15
	} else if (*lwork) < (*m) {
		(*info) = -17
	} else {
		(*info) = 0
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGSVJ1"), -(*info))
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
	//     LARGE = BIG / SQRT( DBLE( M*N ) )
	bigtheta = one / rooteps
	roottol = math.Sqrt(*tol)

	//     .. Initialize the right singular vector matrix ..
	//
	//     RSVEC = LSAME( JOBV, 'Y' )
	emptsw = (*n1) * ((*n) - (*n1))
	notrot = 0

	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	kbl = min(int(8), *n)
	nblr = (*n1) / kbl
	if (nblr * kbl) != (*n1) {
		nblr = nblr + 1
	}
	//     .. the tiling is nblr-by-nblc [tiles]
	nblc = ((*n) - (*n1)) / kbl
	if (nblc * kbl) != ((*n) - (*n1)) {
		nblc = nblc + 1
	}
	blskip = pow(kbl, 2) + 1
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.
	rowskip = min(5, kbl)
	//[TP] ROWSKIP is a tuning parameter.
	swband = 0
	//[TP] SWBAND is a tuning parameter. It is meaningful and effective
	//     if ZGESVJ is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm ZGEJSV.
	//
	//
	//     | *   *   * [x] [x] [x]|
	//     | *   *   * [x] [x] [x]|    Row-cycling in the nblr-by-nblc [x] blocks.
	//     | *   *   * [x] [x] [x]|    Row-cyclic pivoting inside each [x] block.
	//     |[x] [x] [x] *   *   * |
	//     |[x] [x] [x] *   *   * |
	//     |[x] [x] [x] *   *   * |

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
		for ibr = 1; ibr <= nblr; ibr++ {

			igl = (ibr-1)*kbl + 1

			// ... go to the off diagonal blocks
			igl = (ibr-1)*kbl + 1

			//            DO 2010 jbc = ibr + 1, NBL
			for jbc = 1; jbc <= nblc; jbc++ {

				jgl = (jbc-1)*kbl + (*n1) + 1

				//        doing the block at ( ibr, jbc )
				ijblsk = 0
				for p = igl; p <= min(igl+kbl-1, *n1); p++ {

					aapp = sva.Get(p - 1)
					if aapp > zero {

						pskipped = 0

						for q = jgl; q <= min(jgl+kbl-1, *n); q++ {

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
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1, 1), a.CVector(0, q-1, 1)) / complex(aaqq, 0)) / complex(aapp, 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, p-1, 1), work.Off(0, 1))
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, work.Off(0, 1), a.CVector(0, q-1, 1)) / complex(aaqq, 0)
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (goblas.Zdotc(*m, a.CVector(0, p-1, 1), a.CVector(0, q-1, 1)) / complex(math.Max(aaqq, aapp), 0)) / complex(math.Min(aaqq, aapp), 0)
									} else {
										goblas.Zcopy(*m, a.CVector(0, q-1, 1), work.Off(0, 1))
										Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
										aapq = goblas.Zdotc(*m, a.CVector(0, p-1, 1), work.Off(0, 1)) / complex(aapp, 0)
									}
								}

								//                           AAPQ = AAPQ * CONJG(CWORK(p))*CWORK(q)
								aapq1 = -cmplx.Abs(aapq)
								mxaapq = math.Max(mxaapq, -aapq1)

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
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq1))
											mxsinj = math.Max(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq1)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = math.Max(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq1)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq1))

											Zrot(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), a.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											if rsvec {
												Zrot(&mvl, v.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, q-1), func() *int { y := 1; return &y }(), &cs, toPtrc128(cmplx.Conj(ompq)*complex(sn, 0)))
											}
										}
										d.Set(p-1, -d.Get(q-1)*ompq)

									} else {
										//              .. have to use modified Gram-Schmidt like transformation
										if aapp > aaqq {
											goblas.Zcopy(*m, a.CVector(0, p-1, 1), work.Off(0, 1))
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											goblas.Zaxpy(*m, -aapq, work.Off(0, 1), a.CVector(0, q-1, 1))
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aaqq, m, func() *int { y := 1; return &y }(), a.Off(0, q-1), lda, &ierr)
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one-aapq1*aapq1)))
											mxsinj = math.Max(mxsinj, *sfmin)
										} else {
											goblas.Zcopy(*m, a.CVector(0, q-1, 1), work.Off(0, 1))
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aaqq, &one, m, func() *int { y := 1; return &y }(), work.CMatrix(*lda, opts), lda, &ierr)
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &one, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											goblas.Zaxpy(*m, -cmplx.Conj(aapq), work.Off(0, 1), a.CVector(0, p-1, 1))
											Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, &aapp, m, func() *int { y := 1; return &y }(), a.Off(0, p-1), lda, &ierr)
											sva.Set(p-1, aapp*math.Sqrt(math.Max(zero, one-aapq1*aapq1)))
											mxsinj = math.Max(mxsinj, *sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q), SVA(p)
									//           .. recompute SVA(q), SVA(p)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dznrm2(*m, a.CVector(0, q-1, 1)))
										} else {
											t = zero
											aaqq = one
											Zlassq(m, a.CVector(0, q-1), func() *int { y := 1; return &y }(), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dznrm2(*m, a.CVector(0, p-1, 1))
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
							notrot = notrot + min(jgl+kbl-1, *n) - jgl + 1
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
			for p = igl; p <= min(igl+kbl-1, *n); p++ {
				sva.Set(p-1, sva.GetMag(p-1))
			}
			//**
		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get((*n)-1) < rootbig) && (sva.Get((*n)-1) > rootsfmin) {
			sva.Set((*n)-1, goblas.Dznrm2(*m, a.CVector(0, (*n)-1, 1)))
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
		q = goblas.Idamax((*n)-p+1, sva.Off(p-1, 1)) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			aapq = d.Get(p - 1)
			d.Set(p-1, d.Get(q-1))
			d.Set(q-1, aapq)
			goblas.Zswap(*m, a.CVector(0, p-1, 1), a.CVector(0, q-1, 1))
			if rsvec {
				goblas.Zswap(mvl, v.CVector(0, p-1, 1), v.CVector(0, q-1, 1))
			}
		}
	}
}
