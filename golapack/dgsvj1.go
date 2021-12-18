package golapack

import (
	"fmt"
	"math"

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
// Dgsvj1 applies few sweeps of Jacobi rotations in the column space of
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
func Dgsvj1(jobv byte, m, n, n1 int, a *mat.Matrix, d, sva *mat.Vector, mv int, v *mat.Matrix, eps, sfmin, tol float64, nsweep int, work *mat.Vector, lwork int) (info int, err error) {
	var applv, rotok, rsvec bool
	var aapp, aapp0, aapq, aaqq, apoaq, aqoap, big, bigtheta, cs, half, mxaapq, mxsinj, one, rootbig, rooteps, rootsfmin, roottol, small, sn, t, temp1, theta, thsign, zero float64
	var blskip, emptsw, i, ibr, igl, ijblsk, iswrot, jbc, jgl, kbl, mvl, nblc, nblr, notrot, p, pskipped, q, rowskip, swband int

	fastr := mat.NewDrotMatrix()

	zero = 0.0
	half = 0.5
	one = 1.0

	//     Test the input parameters.
	applv = jobv == 'A'
	rsvec = jobv == 'V'
	if !(rsvec || applv || jobv == 'N') {
		err = fmt.Errorf("!(rsvec || applv || jobv == 'N'): jobv='%c'", jobv)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if (n < 0) || (n > m) {
		err = fmt.Errorf("(n < 0) || (n > m): m=%v, n=%v", m, n)
	} else if n1 < 0 {
		err = fmt.Errorf("n1 < 0: n1=%v", n1)
	} else if a.Rows < m {
		err = fmt.Errorf("a.Rows < m: a.Rows=%v, m=%v", a.Rows, m)
	} else if (rsvec || applv) && (mv < 0) {
		err = fmt.Errorf("(rsvec || applv) && (mv < 0): jobv='%c', mv=%v", jobv, mv)
	} else if (rsvec && (v.Rows < n)) || (applv && (v.Rows < mv)) {
		err = fmt.Errorf("(rsvec && (v.Rows < n)) || (applv && (v.Rows < mv)): jobv='%c', v.Rows=%v, mv=%v", jobv, v.Rows, mv)
	} else if tol <= eps {
		err = fmt.Errorf("tol <= eps: tol=%v, eps=%v", tol, eps)
	} else if nsweep < 0 {
		err = fmt.Errorf("nsweep < 0: nsweep=%v", nsweep)
	} else if lwork < m {
		err = fmt.Errorf("lwork < m: lwork=%v, m=%v", lwork, m)
	}

	//     #:(
	if err != nil {
		gltest.Xerbla2("Dgsvj1", err)
		return
	}

	if rsvec {
		mvl = n
	} else if applv {
		mvl = mv
	}
	rsvec = rsvec || applv
	rooteps = math.Sqrt(eps)
	rootsfmin = math.Sqrt(sfmin)
	small = sfmin / eps
	big = one / sfmin
	rootbig = one / rootsfmin
	// large = big / math.Sqrt(float64(m*n))
	bigtheta = one / rooteps
	roottol = math.Sqrt(tol)

	//     .. Initialize the right singular vector matrix ..
	//
	//     RSVEC = LSAME( JOBV, 'Y' )
	emptsw = n1 * (n - n1)
	notrot = 0
	fastr.Flag = int(zero)

	//     .. Row-cyclic pivot strategy with de Rijk's pivoting ..
	kbl = min(8, n)
	nblr = n1 / kbl
	if (nblr * kbl) != n1 {
		nblr = nblr + 1
	}
	//     .. the tiling is nblr-by-nblc [tiles]
	nblc = (n - n1) / kbl
	if (nblc * kbl) != (n - n1) {
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

	for i = 1; i <= nsweep; i++ {
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
				jgl = n1 + (jbc-1)*kbl + 1
				//        doing the block at ( ibr, jbc )
				ijblsk = 0
				for p = igl; p <= min(igl+kbl-1, n1); p++ {
					aapp = sva.Get(p - 1)
					if aapp > zero {
						pskipped = 0
						for q = jgl; q <= min(jgl+kbl-1, n); q++ {

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
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										work.Copy(m, a.Off(0, p-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aapp, d.Get(p-1), m, 1, work.Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, q-1).Vector().Dot(m, work, 1, 1) * d.Get(q-1) / aaqq
									}
								} else {
									if aapp >= aaqq {
										rotok = aapp <= (aaqq / small)
									} else {
										rotok = aaqq <= (aapp / small)
									}
									if aapp > (small / aaqq) {
										aapq = (a.Off(0, q-1).Vector().Dot(m, a.Off(0, p-1).Vector(), 1, 1) * d.Get(p-1) * d.Get(q-1) / aaqq) / aapp
									} else {
										a.Off(0, q-1).Vector().Copy(m, a.Off(0, q-1).Vector(), 1, 1)
										if err = Dlascl('G', 0, 0, aaqq, d.Get(q-1), m, 1, work.Matrix(a.Rows, opts)); err != nil {
											panic(err)
										}
										aapq = a.Off(0, p-1).Vector().Dot(m, work, 1, 1) * d.Get(p-1) / aapp
									}
								}
								mxaapq = math.Max(mxaapq, math.Abs(aapq))
								//        TO rotate or NOT to rotate, THAT is the question ...

								if math.Abs(aapq) > tol {
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
											a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
											if rsvec {
												v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
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
													a.Off(0, q-1).Vector().Rotm(m, a.Off(0, p-1).Vector(), 1, 1, fastr)
													if rsvec {
														v.Off(0, q-1).Vector().Rotm(mvl, v.Off(0, p-1).Vector(), 1, 1, fastr)
													}
												} else {
													a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													if rsvec {
														v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
													}
													d.Set(p-1, d.Get(p-1)*cs)
													d.Set(q-1, d.Get(q-1)/cs)
												}
											} else {
												if d.Get(q-1) >= one {
													a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
													a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
													if rsvec {
														v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
													}
													d.Set(p-1, d.Get(p-1)/cs)
													d.Set(q-1, d.Get(q-1)*cs)
												} else {
													if d.Get(p-1) >= d.Get(q-1) {
														a.Off(0, p-1).Vector().Axpy(m, -t*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														a.Off(0, q-1).Vector().Axpy(m, cs*sn*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														d.Set(p-1, d.Get(p-1)*cs)
														d.Set(q-1, d.Get(q-1)/cs)
														if rsvec {
															v.Off(0, p-1).Vector().Axpy(mvl, -t*aqoap, v.Off(0, q-1).Vector(), 1, 1)
															v.Off(0, q-1).Vector().Axpy(mvl, cs*sn*apoaq, v.Off(0, p-1).Vector(), 1, 1)
														}
													} else {
														a.Off(0, q-1).Vector().Axpy(m, t*apoaq, a.Off(0, p-1).Vector(), 1, 1)
														a.Off(0, p-1).Vector().Axpy(m, -cs*sn*aqoap, a.Off(0, q-1).Vector(), 1, 1)
														d.Set(p-1, d.Get(p-1)/cs)
														d.Set(q-1, d.Get(q-1)*cs)
														if rsvec {
															v.Off(0, q-1).Vector().Axpy(mvl, t*apoaq, v.Off(0, p-1).Vector(), 1, 1)
															v.Off(0, p-1).Vector().Axpy(mvl, -cs*sn*aqoap, v.Off(0, q-1).Vector(), 1, 1)
														}
													}
												}
											}
										}
									} else {
										if aapp > aaqq {
											work.Copy(m, a.Off(0, p-1).Vector(), 1, 1)
											if err = Dlascl('G', 0, 0, aapp, one, m, 1, work.Matrix(a.Rows, opts)); err != nil {
												panic(err)
											}
											if err = Dlascl('G', 0, 0, aaqq, one, m, 1, a.Off(0, q-1)); err != nil {
												panic(err)
											}
											temp1 = -aapq * d.Get(p-1) / d.Get(q-1)
											a.Off(0, q-1).Vector().Axpy(m, temp1, work, 1, 1)
											if err = Dlascl('G', 0, 0, one, aaqq, m, 1, a.Off(0, q-1)); err != nil {
												panic(err)
											}
											sva.Set(q-1, aaqq*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, sfmin)
										} else {
											work.Copy(m, a.Off(0, q-1).Vector(), 1, 1)
											if err = Dlascl('G', 0, 0, aaqq, one, m, 1, work.Matrix(a.Rows, opts)); err != nil {
												panic(err)
											}
											if err = Dlascl('G', 0, 0, aapp, one, m, 1, a.Off(0, p-1)); err != nil {
												panic(err)
											}
											temp1 = -aapq * d.Get(q-1) / d.Get(p-1)
											a.Off(0, p-1).Vector().Axpy(m, temp1, work, 1, 1)
											if err = Dlascl('G', 0, 0, one, aapp, m, 1, a.Off(0, p-1)); err != nil {
												panic(err)
											}
											sva.Set(p-1, aapp*math.Sqrt(math.Max(zero, one-aapq*aapq)))
											mxsinj = math.Max(mxsinj, sfmin)
										}
									}
									//           END IF ROTOK THEN ... ELSE
									//
									//           In the case of cancellation in updating SVA(q)
									//           .. recompute SVA(q)
									if math.Pow(sva.Get(q-1)/aaqq, 2) <= rooteps {
										if (aaqq < rootbig) && (aaqq > rootsfmin) {
											sva.Set(q-1, a.Off(0, q-1).Vector().Nrm2(m, 1)*d.Get(q-1))
										} else {
											t = zero
											aaqq = one
											t, aaqq = Dlassq(m, a.Off(0, q-1).Vector(), 1, t, aaqq)
											sva.Set(q-1, t*math.Sqrt(aaqq)*d.Get(q-1))
										}
									}
									if math.Pow(aapp/aapp0, 2) <= rooteps {
										if (aapp < rootbig) && (aapp > rootsfmin) {
											aapp = a.Off(0, p-1).Vector().Nrm2(m, 1) * d.Get(p-1)
										} else {
											t = zero
											aapp = one
											t, aapp = Dlassq(m, a.Off(0, p-1).Vector(), 1, t, aapp)
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
							notrot = notrot + min(jgl+kbl-1, n) - jgl + 1
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
			for p = igl; p <= min(igl+kbl-1, n); p++ {
				sva.Set(p-1, math.Abs(sva.Get(p-1)))
			}
			//**   IF ( NOTROT .GE. EMPTSW ) GO TO 1994
		}
		//2000 :: end of the ibr-loop
		//
		//     .. update SVA(N)
		if (sva.Get(n-1) < rootbig) && (sva.Get(n-1) > rootsfmin) {
			sva.Set(n-1, a.Off(0, n-1).Vector().Nrm2(m, 1)*d.Get(n-1))
		} else {
			t = zero
			aapp = one
			t, aapp = Dlassq(m, a.Off(0, n-1).Vector(), 1, t, aapp)
			sva.Set(n-1, t*math.Sqrt(aapp)*d.Get(n-1))
		}

		//     Additional steering devices
		if (i < swband) && ((mxaapq <= roottol) || (iswrot <= n)) {
			swband = i
		}
		if (i > swband+1) && (mxaapq < float64(n)*tol) && (float64(n)*mxaapq*mxsinj < tol) {
			goto label1994
		}

		if notrot >= emptsw {
			goto label1994
		}
	}
	//     end i=1:NSWEEP loop
	// #:) Reaching this point means that the procedure has completed the given
	//     number of sweeps.
	info = nsweep - 1
	goto label1995
label1994:
	;
	// #:) Reaching this point means that during the i-th sweep all pivots were
	//     below the given threshold, causing early exit.
	info = 0
	// #:) INFO = 0 confirms successful iterations.
label1995:
	;

	//     Sort the vector D
	for p = 1; p <= n-1; p++ {
		q = sva.Off(p-1).Iamax(n-p+1, 1) + p - 1
		if p != q {
			temp1 = sva.Get(p - 1)
			sva.Set(p-1, sva.Get(q-1))
			sva.Set(q-1, temp1)
			temp1 = d.Get(p - 1)
			d.Set(p-1, d.Get(q-1))
			d.Set(q-1, temp1)
			a.Off(0, q-1).Vector().Swap(m, a.Off(0, p-1).Vector(), 1, 1)
			if rsvec {
				v.Off(0, q-1).Vector().Swap(mvl, v.Off(0, p-1).Vector(), 1, 1)
			}
		}
	}

	return
}
