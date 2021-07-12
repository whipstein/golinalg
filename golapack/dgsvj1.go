package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgsvj1 is called from DGESVJ as a pre-processor and that is its main
// purpose. It applies Jacobi rotations in the same way as DGESVJ does, but
// it targets only particular pivots and it does not check convergence
// (stopping criterion). Few tunning parameters (marked by [TP]) are
// available for the implementer.
//
// Further Details
// ~~~~~~~~~~~~~~~
// DGSVJ1 applies few sweeps of Jacobi rotations in the column space of
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
func Dgsvj1(jobv byte, m, n, n1 *int, a *mat.Matrix, lda *int, d, sva *mat.Vector, mv *int, v *mat.Matrix, ldv *int, eps, sfmin, tol *float64, nsweep *int, work *mat.Vector, lwork, info *int) {
	var applv, rotok, rsvec bool
	var aapp, aapp0, aapq, aaqq, apoaq, aqoap, big, bigtheta, cs, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, small, sn, t, temp1, theta, thsign, zero float64
	var blskip, emptsw, i, ibr, ierr, igl, ijblsk, iswrot, jbc, jgl, kbl, mvl, nblc, nblr, notrot, p, pskipped, q, rowskip, swband int

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

	//     #:(
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGSVJ1"), -(*info))
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
	// large = big / math.Sqrt(float64((*m)*(*n)))
	bigtheta = one / rooteps
	roottol = math.Sqrt(*tol)

	//     .. Initialize the right singular vector matrix ..
	//
	//     RSVEC = LSAME( JOBV, 'Y' )
	emptsw = (*n1) * ((*n) - (*n1))
	notrot = 0
	fastr.Flag = int(zero)

	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	kbl = min(8, *n)
	nblr = (*n1) / kbl
	if (nblr * kbl) != (*n1) {
		nblr = nblr + 1
	}
	//     .. the tiling is nblr-by-nblc [tiles]
	nblc = ((*n) - (*n1)) / kbl
	if (nblc * kbl) != ((*n) - (*n1)) {
		nblc = nblc + 1
	}
	blskip = int(math.Pow(float64(kbl), 2)) + 1
	//[TP] BLKSKIP is a tuning parameter that depends on SWBAND and KBL.
	rowskip = min(5, kbl)
	//[TP] ROWSKIP is a tuning parameter.
	swband = 0
	//[TP] SWBAND is a tuning parameter. It is meaningful and effective
	//     if SGESVJ is used as a computational routine in the preconditioned
	//     Jacobi SVD algorithm SGESVJ.
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

		for ibr = 1; ibr <= nblr; ibr++ {
			igl = (ibr-1)*kbl + 1

			//........................................................
			// ... go to the off diagonal blocks
			igl = (ibr-1)*kbl + 1
			for jbc = 1; jbc <= nblc; jbc++ {
				jgl = (*n1) + (jbc-1)*kbl + 1
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
								//        .. Safe Gram matrix computation ..
								if aaqq >= one {
									if aapp >= aaqq {
										rotok = (small * aapp) <= aaqq
									} else {
										rotok = (small * aaqq) <= aapp
									}
									if aapp < (big / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1)) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, p-1, 1), work.Off(0, 1))
										Dlascl('G', toPtr(0), toPtr(0), &aapp, d.GetPtr(p-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, a.Vector(0, q-1, 1)) * d.Get(q-1) / aaqq
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (goblas.Ddot(*m, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1)) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										goblas.Dcopy(*m, a.Vector(0, q-1, 1), work)
										Dlascl('G', toPtr(0), toPtr(0), &aaqq, d.GetPtr(q-1), m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
										aapq = goblas.Ddot(*m, work, a.Vector(0, p-1, 1)) * d.Get(p-1) / aapp
									}
								}
								mxaapq = math.Max(mxaapq, math.Abs(aapq))
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
											goblas.Drotm(*m, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1), fastr)
											if rsvec {
												goblas.Drotm(mvl, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1), fastr)
											}
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))
											mxsinj = math.Max(mxsinj, math.Abs(t))
										} else {
											//                 .. choose correct signum for THETA and rotate
											thsign = -math.Copysign(one, aapq)
											if aaqq > aapp0 {
												thsign = -thsign
											}
											t = one / (theta + thsign*math.Sqrt(one+theta*theta))
											cs = math.Sqrt(one / (one + t*t))
											sn = t * cs
											mxsinj = math.Max(mxsinj, math.Abs(sn))
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one+t*apoaq*aapq)))
											aapp = aapp * math.Sqrt(math.Max(zero, one-t*aqoap*aapq))
											apoaq = d.Get(p-1) / d.Get(q-1)
											aqoap = d.Get(q-1) / d.Get(p-1)
											if d.Get(p-1) >= one {

												if d.Get(q-1) >= one {
													fastr.H21 = t * apoaq
													fastr.H12 = -t * aqoap
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)*cs)
													goblas.Drotm(*m, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1), fastr)
													if rsvec {
														goblas.Drotm(mvl, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1), fastr)
													}
												} else {
													goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1, 1), a.Vector(0, p-1, 1))
													goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1))
													if rsvec {
														goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1, 1), v.Vector(0, p-1, 1))
														goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1))
													}
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)/cs)
												}
											} else {
												if d.Get(q-1) >= one {
													goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1))
													goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1, 1), a.Vector(0, p-1, 1))
													if rsvec {
														goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1))
														goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1, 1), v.Vector(0, p-1, 1))
													}
													d.Set(p-1, d.Get(p-1)/cs)
													d.Set(q-1, d.Get(q-1)*cs)
												} else {
													if d.Get(p-1) >= d.Get(q-1) {
														goblas.Daxpy(*m, -t*aqoap, a.Vector(0, q-1, 1), a.Vector(0, p-1, 1))
														goblas.Daxpy(*m, cs*sn*apoaq, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1))
														d.Set(p-1, d.Get(p-1)*cs)
														d.Set(q-1, d.Get(q-1)/cs)
														if rsvec {
															goblas.Daxpy(mvl, -t*aqoap, v.Vector(0, q-1, 1), v.Vector(0, p-1, 1))
															goblas.Daxpy(mvl, cs*sn*apoaq, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1))
														}
													} else {
														goblas.Daxpy(*m, t*apoaq, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1))
														goblas.Daxpy(*m, -cs*sn*aqoap, a.Vector(0, q-1, 1), a.Vector(0, p-1, 1))
														d.Set(p-1, d.Get(p-1)/cs)
														d.Set(q-1, d.Get(q-1)*cs)
														if rsvec {
															goblas.Daxpy(mvl, t*apoaq, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1))
															goblas.Daxpy(mvl, -cs*sn*aqoap, v.Vector(0, q-1, 1), v.Vector(0, p-1, 1))
														}
													}
												}
											}
										}
									} else {
										if aapp > aaqq {
											goblas.Dcopy(*m, a.Vector(0, p-1, 1), work)
											Dlascl('G', toPtr(0), toPtr(0), &aapp, &one, m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
											Dlascl('G', toPtr(0), toPtr(0), &aaqq, &one, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
											temp1 = -aapq * d.Get(p-1) / d.Get(q-1)
											goblas.Daxpy(*m, temp1, work, a.Vector(0, q-1, 1))
											Dlascl('G', toPtr(0), toPtr(0), &one, &aaqq, m, toPtr(1), a.Off(0, q-1), lda, &ierr)
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, *sfmin)
										} else {
											goblas.Dcopy(*m, a.Vector(0, q-1, 1), work)
											Dlascl('G', toPtr(0), toPtr(0), &aaqq, &one, m, toPtr(1), work.Matrix(*lda, opts), lda, &ierr)
											Dlascl('G', toPtr(0), toPtr(0), &aapp, &one, m, toPtr(1), a.Off(0, p-1), lda, &ierr)
											temp1 = -aapq * d.Get(q-1) / d.Get(p-1)
											goblas.Daxpy(*m, temp1, work, a.Vector(0, p-1, 1))
											Dlascl('G', toPtr(0), toPtr(0), &one, &aapp, m, toPtr(1), a.Off(0, p-1), lda, &ierr)
											sva.Set(p-1, aapp*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, *sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q)
									//           .. recompute SVA(q)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, goblas.Dnrm2(*m, a.Vector(0, q-1, 1))*d.Get(q-1))
										} else {
											t = zero
											aaqq = one
											Dlassq(m, a.Vector(0, q-1), toPtr(1), &t, &aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*d.Get(q-1))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = goblas.Dnrm2(*m, a.Vector(0, p-1, 1)) * d.Get(p-1)
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
									//           SKIPPED  = SKIPPED  + 1
									pskipped = pskipped + 1
									ijblsk = ijblsk + 1
								}
							} else {
								notrot = notrot + 1
								pskipped = pskipped + 1
								ijblsk = ijblsk + 1
							}
							//      IF ( NOTROT .GE. EMPTSW )  GO TO 2011
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
						//**      IF ( NOTROT .GE. EMPTSW )  GO TO 2011
					}
				}
				//     end of the p-loop
			}
			//     end of the jbc-loop
		label2011:
			;
			//2011 bailed out of the jbc-loop
			for p = igl; p <= min(igl+kbl-1, *n); p++ {
				sva.Set(p-1, math.Abs(sva.Get(p-1)))
			}
			//**   IF ( NOTROT .GE. EMPTSW ) GO TO 1994
		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get((*n)-1) < rootbig) && (sva.Get((*n)-1) > rootsfmin) {
			sva.Set((*n)-1, goblas.Dnrm2(*m, a.Vector(0, (*n)-1, 1))*d.Get((*n)-1))
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
	//     number of sweeps.
	(*info) = (*nsweep) - 1
	goto label1995
label1994:
	;
	// #:) Reaching this point means that during the i-th sweep all pivots were
	//     below the given threshold, causing early exit.
	(*info) = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the vector D
	for p = 1; p <= (*n)-1; p++ {
		q = goblas.Idamax((*n)-p+1, sva.Off(p-1)) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			temp1 = d.Get(p - 1)
			d.Set(p-1, d.Get(q-1))
			d.Set(q-1, temp1)
			goblas.Dswap(*m, a.Vector(0, p-1, 1), a.Vector(0, q-1, 1))
			if rsvec {
				goblas.Dswap(mvl, v.Vector(0, p-1, 1), v.Vector(0, q-1, 1))
			}
		}
	}
}
