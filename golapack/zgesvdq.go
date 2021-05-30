package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zgesvdq computes the singular value decomposition (SVD) of a complex
// M-by-N matrix A, where M >= N. The SVD of A is written as
//                                    [++]   [xx]   [x0]   [xx]
//              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
//                                    [++]   [xx]
// where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
// matrix, and V is an N-by-N unitary matrix. The diagonal elements
// of SIGMA are the singular values of A. The columns of U and V are the
// left and the right singular vectors of A, respectively.
func Zgesvdq(joba, jobp, jobr, jobu, jobv byte, m, n *int, a *mat.CMatrix, lda *int, s *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv, numrank *int, iwork *[]int, liwork *int, cwork *mat.CVector, lcwork *int, rwork *mat.Vector, lrwork, info *int) {
	var accla, acclh, acclm, ascaled, conda, dntwu, dntwv, lquery, lsvc0, lsvec, rowprm, rsvec, rtrans, wntua, wntuf, wntur, wntus, wntva, wntvr bool
	var cone, ctmp, czero complex128
	var big, epsln, one, rtmp, sconda, sfmin, zero float64
	var ierr, iminwrk, lwcon, lwlqf, lwqp3, lwqrf, lwrkZgelqf, lwrkZgeqp3, lwrkZgeqrf, lwrkZgesvd, lwrkZgesvd2, lwrkZunmlq, lwrkZunmqr, lwrkZunmqr2, lwsvd, lwsvd2, lwunlq, lwunq, lwunq2, minwrk, minwrk2, n1, nr, optratio, optwrk, optwrk2, p, q, rminwrk int
	cdummy := cvf(1)
	rdummy := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input arguments
	wntus = jobu == 'S' || jobu == 'U'
	wntur = jobu == 'R'
	wntua = jobu == 'A'
	wntuf = jobu == 'F'
	lsvc0 = wntus || wntur || wntua
	lsvec = lsvc0 || wntuf
	dntwu = jobu == 'N'

	wntvr = jobv == 'R'
	wntva = jobv == 'A' || jobv == 'V'
	rsvec = wntvr || wntva
	dntwv = jobv == 'N'

	accla = joba == 'A'
	acclm = joba == 'M'
	conda = joba == 'E'
	acclh = joba == 'H' || conda

	rowprm = jobp == 'P'
	rtrans = jobr == 'T'

	if rowprm {
		iminwrk = maxint(1, (*n)+(*m)-1)
		rminwrk = maxint(2, *m, 5*(*n))
	} else {
		iminwrk = maxint(1, *n)
		rminwrk = maxint(2, 5*(*n))
	}
	lquery = ((*liwork) == -1 || (*lcwork) == -1 || (*lrwork) == -1)
	(*info) = 0
	if !(accla || acclm || acclh) {
		(*info) = -1
	} else if !(rowprm || jobp == 'N') {
		(*info) = -2
	} else if !(rtrans || jobr == 'N') {
		(*info) = -3
	} else if !(lsvec || dntwu) {
		(*info) = -4
	} else if wntur && wntva {
		(*info) = -5
	} else if !(rsvec || dntwv) {
		(*info) = -5
	} else if (*m) < 0 {
		(*info) = -6
	} else if ((*n) < 0) || ((*n) > (*m)) {
		(*info) = -7
	} else if (*lda) < maxint(1, *m) {
		(*info) = -9
	} else if (*ldu) < 1 || (lsvc0 && (*ldu) < (*m)) || (wntuf && (*ldu) < (*n)) {
		(*info) = -12
	} else if (*ldv) < 1 || (rsvec && (*ldv) < (*n)) || (conda && (*ldv) < (*n)) {
		(*info) = -14
	} else if (*liwork) < iminwrk && !lquery {
		(*info) = -17
	}

	if (*info) == 0 {
		//        .. compute the minimal and the optimal workspace lengths
		//        [[The expressions for computing the minimal and the optimal
		//        values of LCWORK are written with a lot of redundancy and
		//        can be simplified. However, this detailed form is easier for
		//        maintenance and modifications of the code.]]
		//
		//        .. minimal workspace length for ZGEQP3 of an M x N matrix
		lwqp3 = (*n) + 1
		//        .. minimal workspace length for ZUNMQR to build left singular vectors
		if wntus || wntur {
			lwunq = maxint(*n, 1)
		} else if wntua {
			lwunq = maxint(*m, 1)
		}
		//        .. minimal workspace length for ZPOCON of an N x N matrix
		lwcon = 2 * (*n)
		//        .. ZGESVD of an N x N matrix
		lwsvd = maxint(3*(*n), 1)
		if lquery {
			Zgeqp3(m, n, a, lda, iwork, cdummy, cdummy, toPtr(-1), rdummy, &ierr)
			lwrkZgeqp3 = int(cdummy.GetRe(0))
			if wntus || wntur {
				Zunmqr('L', 'N', m, n, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
				lwrkZunmqr = int(cdummy.GetRe(0))
			} else if wntua {
				Zunmqr('L', 'N', m, m, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
				lwrkZunmqr = int(cdummy.GetRe(0))
			} else {
				lwrkZunmqr = 0
			}
		}
		minwrk = 2
		optwrk = 2
		if !(lsvec || rsvec) {
			//            .. minimal and optimal sizes of the complex workspace if
			//            only the singular values are requested
			if conda {
				minwrk = maxint((*n)+lwqp3, lwcon, lwsvd)
			} else {
				minwrk = maxint((*n)+lwqp3, lwsvd)
			}
			if lquery {
				Zgesvd('N', 'N', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
				lwrkZgesvd = int(cdummy.GetRe(0))
				if conda {
					optwrk = maxint((*n)+lwrkZgeqp3, (*n)+lwcon, lwrkZgesvd)
				} else {
					optwrk = maxint((*n)+lwrkZgeqp3, lwrkZgesvd)
				}
			}
		} else if lsvec && (!rsvec) {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            singular values and the left singular vectors are requested
			if conda {
				minwrk = (*n) + maxint(lwqp3, lwcon, lwsvd, lwunq)
			} else {
				minwrk = (*n) + maxint(lwqp3, lwsvd, lwunq)
			}
			if lquery {
				if rtrans {
					Zgesvd('N', 'O', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
				} else {
					Zgesvd('O', 'N', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
				}
				lwrkZgesvd = int(cdummy.GetRe(0))
				if conda {
					optwrk = (*n) + maxint(lwrkZgeqp3, lwcon, lwrkZgesvd, lwrkZunmqr)
				} else {
					optwrk = (*n) + maxint(lwrkZgeqp3, lwrkZgesvd, lwrkZunmqr)
				}
			}
		} else if rsvec && (!lsvec) {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            singular values and the right singular vectors are requested
			if conda {
				minwrk = (*n) + maxint(lwqp3, lwcon, lwsvd)
			} else {
				minwrk = (*n) + maxint(lwqp3, lwsvd)
			}
			if lquery {
				if rtrans {
					Zgesvd('O', 'N', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
				} else {
					Zgesvd('N', 'O', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
				}
				lwrkZgesvd = int(cdummy.GetRe(0))
				if conda {
					optwrk = (*n) + maxint(lwrkZgeqp3, lwcon, lwrkZgesvd)
				} else {
					optwrk = (*n) + maxint(lwrkZgeqp3, lwrkZgesvd)
				}
			}
		} else {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            full SVD is requested
			if rtrans {
				minwrk = maxint(lwqp3, lwsvd, lwunq)
				if conda {
					minwrk = maxint(minwrk, lwcon)
				}
				minwrk = minwrk + (*n)
				if wntva {
					//                   .. minimal workspace length for N x N/2 ZGEQRF
					lwqrf = maxint((*n)/2, 1)
					//                   .. minimal workspace lengt for N/2 x N/2 ZGESVD
					lwsvd2 = maxint(3*((*n)/2), 1)
					lwunq2 = maxint(*n, 1)
					minwrk2 = maxint(lwqp3, (*n)/2+lwqrf, (*n)/2+lwsvd2, (*n)/2+lwunq2, lwunq)
					if conda {
						minwrk2 = maxint(minwrk2, lwcon)
					}
					minwrk2 = (*n) + minwrk2
					minwrk = maxint(minwrk, minwrk2)
				}
			} else {
				minwrk = maxint(lwqp3, lwsvd, lwunq)
				if conda {
					minwrk = maxint(minwrk, lwcon)
				}
				minwrk = minwrk + (*n)
				if wntva {
					//                   .. minimal workspace length for N/2 x N ZGELQF
					lwlqf = maxint((*n)/2, 1)
					lwsvd2 = maxint(3*((*n)/2), 1)
					lwunlq = maxint(*n, 1)
					minwrk2 = maxint(lwqp3, (*n)/2+lwlqf, (*n)/2+lwsvd2, (*n)/2+lwunlq, lwunq)
					if conda {
						minwrk2 = maxint(minwrk2, lwcon)
					}
					minwrk2 = (*n) + minwrk2
					minwrk = maxint(minwrk, minwrk2)
				}
			}
			if lquery {
				if rtrans {
					Zgesvd('O', 'A', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
					lwrkZgesvd = int(cdummy.GetRe(0))
					optwrk = maxint(lwrkZgeqp3, lwrkZgesvd, lwrkZunmqr)
					if conda {
						optwrk = maxint(optwrk, lwcon)
					}
					optwrk = (*n) + optwrk
					if wntva {
						Zgeqrf(n, toPtr((*n)/2), u, ldu, cdummy, cdummy, toPtr(-1), &ierr)
						lwrkZgeqrf = int(cdummy.GetRe(0))
						Zgesvd('S', 'O', toPtr((*n)/2), toPtr((*n)/2), v, ldv, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
						lwrkZgesvd2 = int(cdummy.GetRe(0))
						Zunmqr('R', 'C', n, n, toPtr((*n)/2), u, ldu, cdummy, v, ldv, cdummy, toPtr(-1), &ierr)
						lwrkZunmqr2 = int(cdummy.GetRe(0))
						optwrk2 = maxint(lwrkZgeqp3, (*n)/2+lwrkZgeqrf, (*n)/2+lwrkZgesvd2, (*n)/2+lwrkZunmqr2)
						if conda {
							optwrk2 = maxint(optwrk2, lwcon)
						}
						optwrk2 = (*n) + optwrk2
						optwrk = maxint(optwrk, optwrk2)
					}
				} else {
					Zgesvd('S', 'O', n, n, a, lda, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
					lwrkZgesvd = int(cdummy.GetRe(0))
					optwrk = maxint(lwrkZgeqp3, lwrkZgesvd, lwrkZunmqr)
					if conda {
						optwrk = maxint(optwrk, lwcon)
					}
					optwrk = (*n) + optwrk
					if wntva {
						Zgelqf(toPtr((*n)/2), n, u, ldu, cdummy, cdummy, toPtr(-1), &ierr)
						lwrkZgelqf = int(cdummy.GetRe(0))
						Zgesvd('S', 'O', toPtr((*n)/2), toPtr((*n)/2), v, ldv, s, u, ldu, v, ldv, cdummy, toPtr(-1), rdummy, &ierr)
						lwrkZgesvd2 = int(cdummy.GetRe(0))
						Zunmlq('R', 'N', n, n, toPtr((*n)/2), u, ldu, cdummy, v, ldv, cdummy, toPtr(-1), &ierr)
						lwrkZunmlq = int(cdummy.GetRe(0))
						optwrk2 = maxint(lwrkZgeqp3, (*n)/2+lwrkZgelqf, (*n)/2+lwrkZgesvd2, (*n)/2+lwrkZunmlq)
						if conda {
							optwrk2 = maxint(optwrk2, lwcon)
						}
						optwrk2 = (*n) + optwrk2
						optwrk = maxint(optwrk, optwrk2)
					}
				}
			}
		}

		minwrk = maxint(2, minwrk)
		optwrk = maxint(2, optwrk)
		if (*lcwork) < minwrk && (!lquery) {
			(*info) = -19
		}

	}

	if (*info) == 0 && (*lrwork) < rminwrk && !lquery {
		(*info) = -21
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZGESVDQ"), -(*info))
		return
	} else if lquery {
		//     Return optimal workspace
		(*iwork)[0] = iminwrk
		cwork.SetRe(0, float64(optwrk))
		cwork.SetRe(1, float64(minwrk))
		rwork.Set(0, float64(rminwrk))
		return
	}

	//     Quick return if the matrix is void.
	if ((*m) == 0) || ((*n) == 0) {
		//     .. all output is void.
		return
	}

	big = Dlamch(Overflow)
	ascaled = false
	if rowprm {
		//           .. reordering the rows in decreasing sequence in the
		//           ell-infinity norm - this enhances numerical robustness in
		//           the case of differently scaled rows.
		for p = 1; p <= (*m); p++ {
			//               RWORK(p) = ABS( A(p,IZAMAX(N,A(p,1),LDA)) )
			//               [[ZLANGE will return NaN if an entry of the p-th row is Nan]]
			rwork.Set(p-1, Zlange('M', func() *int { y := 1; return &y }(), n, a.Off(p-1, 0), lda, rdummy))
			//               .. check for NaN's and Inf's
			if (rwork.Get(p-1) != rwork.Get(p-1)) || ((rwork.Get(p-1) * zero) != zero) {
				(*info) = -8
				gltest.Xerbla([]byte("ZGESVDQ"), -(*info))
				return
			}
		}
		for p = 1; p <= (*m)-1; p++ {
			q = goblas.Idamax(toPtr((*m)-p+1), rwork.Off(p-1), func() *int { y := 1; return &y }()) + p - 1
			(*iwork)[(*n)+p-1] = q
			if p != q {
				rtmp = rwork.Get(p - 1)
				rwork.Set(p-1, rwork.Get(q-1))
				rwork.Set(q-1, rtmp)
			}
		}

		if rwork.Get(0) == zero {
			//              Quick return: A is the M x N zero matrix.
			(*numrank) = 0
			Dlaset('G', n, func() *int { y := 1; return &y }(), &zero, &zero, s.Matrix(*n, opts), n)
			if wntus {
				Zlaset('G', m, n, &czero, &cone, u, ldu)
			}
			if wntua {
				Zlaset('G', m, m, &czero, &cone, u, ldu)
			}
			if wntva {
				Zlaset('G', n, n, &czero, &cone, v, ldv)
			}
			if wntuf {
				Zlaset('G', n, func() *int { y := 1; return &y }(), &czero, &czero, cwork.CMatrix(*n, opts), n)
				Zlaset('G', m, n, &czero, &cone, u, ldu)
			}
			for p = 1; p <= (*n); p++ {
				(*iwork)[p-1] = p
			}
			if rowprm {
				for p = (*n) + 1; p <= (*n)+(*m)-1; p++ {
					(*iwork)[p-1] = p - (*n)
				}
			}
			if conda {
				rwork.Set(0, -1)
			}
			rwork.Set(1, -1)
			return
		}

		if rwork.Get(0) > big/math.Sqrt(float64(*m)) {
			//               .. to prevent overflow in the QR factorization, scale the
			//               matrix by 1/sqrt(M) if too large entry detected
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtrf64(math.Sqrt(float64(*m))), &one, m, n, a, lda, &ierr)
			ascaled = true
		}
		Zlaswp(n, a, lda, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, (*n)+1-1), func() *int { y := 1; return &y }())
	}

	//    .. At this stage, preemptive scaling is done only to avoid column
	//    norms overflows during the QR factorization. The SVD procedure should
	//    have its own scaling to save the singular values from overflows and
	//    underflows. That depends on the SVD procedure.
	if !rowprm {
		rtmp = Zlange('M', m, n, a, lda, rwork)
		if (rtmp != rtmp) || ((rtmp * zero) != zero) {
			(*info) = -8
			gltest.Xerbla([]byte("ZGESVDQ"), -(*info))
			return
		}
		if rtmp > big/math.Sqrt(float64(*m)) {
			//             .. to prevent overflow in the QR factorization, scale the
			//             matrix by 1/sqrt(M) if too large entry detected
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), toPtrf64(math.Sqrt(float64(*m))), &one, m, n, a, lda, &ierr)
			ascaled = true
		}
	}

	//     .. QR factorization with column pivoting
	//
	//     A * P = Q * [ R ]
	//                 [ 0 ]
	for p = 1; p <= (*n); p++ {
		//        .. all columns are free columns
		(*iwork)[p-1] = 0
	}
	Zgeqp3(m, n, a, lda, iwork, cwork, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, &ierr)

	//    If the user requested accuracy level allows truncation in the
	//    computed upper triangular factor, the matrix R is examined and,
	//    if possible, replaced with its leading upper trapezoidal part.
	epsln = Dlamch(Epsilon)
	sfmin = Dlamch(SafeMinimum)
	//     SMALL = SFMIN / EPSLN
	nr = (*n)

	if accla {
		//        Standard absolute error bound suffices. All sigma_i with
		//        sigma_i < N*EPS*||A||_F are flushed to zero. This is an
		//        aggressive enforcement of lower numerical rank by introducing a
		//        backward error of the order of N*EPS*||A||_F.
		nr = 1
		rtmp = math.Sqrt(float64(*n)) * epsln
		for p = 2; p <= (*n); p++ {
			if a.GetMag(p-1, p-1) < (rtmp * a.GetMag(0, 0)) {
				goto label3002
			}
			nr = nr + 1
		}
	label3002:
	} else if acclm {
		//        .. similarly as above, only slightly more gentle (less aggressive).
		//        Sudden drop on the diagonal of R is used as the criterion for being
		//        close-to-rank-deficient. The threshold is set to EPSLN=DLAMCH('E').
		//        [[This can be made more flexible by replacing this hard-coded value
		//        with a user specified threshold.]] Also, the values that underflow
		//        will be truncated.
		nr = 1
		for p = 2; p <= (*n); p++ {
			if (a.GetMag(p-1, p-1) < (epsln * a.GetMag(p-1-1, p-1-1))) || (a.GetMag(p-1, p-1) < sfmin) {
				goto label3402
			}
			nr = nr + 1
		}
	label3402:
	} else {
		//        .. RRQR not authorized to determine numerical rank except in the
		//        obvious case of zero pivots.
		//        .. inspect R for exact zeros on the diagonal;
		//        R(i,i)=0 => R(i:N,i:N)=0.
		nr = 1
		for p = 2; p <= (*n); p++ {
			if a.GetMag(p-1, p-1) == zero {
				goto label3502
			}
			nr = nr + 1
		}
	label3502:
		;

		if conda {
			//           Estimate the scaled condition number of A. Use the fact that it is
			//           the same as the scaled condition number of R.
			//              .. V is used as workspace
			Zlacpy('U', n, n, a, lda, v, ldv)
			//              Only the leading NR x NR submatrix of the triangular factor
			//              is considered. Only if NR=N will this give a reliable error
			//              bound. However, even for NR < N, this can be used on an
			//              expert level and obtain useful information in the sense of
			//              perturbation theory.
			for p = 1; p <= nr; p++ {
				rtmp = goblas.Dznrm2(&p, v.CVector(0, p-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(&p, toPtrf64(one/rtmp), v.CVector(0, p-1), func() *int { y := 1; return &y }())
			}
			if !(lsvec || rsvec) {
				Zpocon('U', &nr, v, ldv, &one, &rtmp, cwork, rwork, &ierr)
			} else {
				Zpocon('U', &nr, v, ldv, &one, &rtmp, cwork.Off((*n)+1-1), rwork, &ierr)
			}
			sconda = one / math.Sqrt(rtmp)
			//           For NR=N, SCONDA is an estimate of SQRT(||(R^* * R)^(-1)||_1),
			//           N^(-1/4) * SCONDA <= ||R^(-1)||_2 <= N^(1/4) * SCONDA
			//           See the reference [1] for more details.
		}

	}

	if wntur {
		n1 = nr
	} else if wntus || wntuf {
		n1 = (*n)
	} else if wntua {
		n1 = (*m)
	}

	if !(rsvec || lsvec) {
		//.......................................................................
		//        .. only the singular values are requested
		//.......................................................................
		if rtrans {
			//         .. compute the singular values of R**H = [A](1:NR,1:N)**H
			//           .. set the lower triangle of [A] to [A](1:NR,1:N)**H and
			//           the upper triangle of [A] to zero.
			for p = 1; p <= minint(*n, nr); p++ {
				a.Set(p-1, p-1, a.GetConj(p-1, p-1))
				for q = p + 1; q <= (*n); q++ {
					a.Set(q-1, p-1, a.GetConj(p-1, q-1))
					if q <= nr {
						a.Set(p-1, q-1, czero)
					}
				}
			}

			Zgesvd('N', 'N', n, &nr, a, lda, s, u, ldu, v, ldv, cwork, lcwork, rwork, info)

		} else {
			//           .. compute the singular values of R = [A](1:NR,1:N)
			if nr > 1 {
				Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, a.Off(1, 0), lda)
			}
			Zgesvd('N', 'N', &nr, n, a, lda, s, u, ldu, v, ldv, cwork, lcwork, rwork, info)

		}

	} else if lsvec && (!rsvec) {
		//.......................................................................
		//       .. the singular values and the left singular vectors requested
		//.......................................................................""""""""
		if rtrans {
			//            .. apply ZGESVD to R**H
			//            .. copy R**H into [U] and overwrite [U] with the right singular
			//            vectors of R
			for p = 1; p <= nr; p++ {
				for q = p; q <= (*n); q++ {
					u.Set(q-1, p-1, a.GetConj(p-1, q-1))
				}
			}
			if nr > 1 {
				Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(0, 1), ldu)
			}
			//           .. the left singular vectors not computed, the NR right singular
			//           vectors overwrite [U](1:NR,1:NR) as conjugate transposed. These
			//           will be pre-multiplied by Q to build the left singular vectors of A.
			Zgesvd('N', 'O', n, &nr, u, ldu, s, u, ldu, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)

			for p = 1; p <= nr; p++ {
				u.Set(p-1, p-1, u.GetConj(p-1, p-1))
				for q = p + 1; q <= nr; q++ {
					ctmp = u.GetConj(q-1, p-1)
					u.Set(q-1, p-1, u.GetConj(p-1, q-1))
					u.Set(p-1, q-1, ctmp)
				}
			}

		} else {
			//            .. apply ZGESVD to R
			//            .. copy R into [U] and overwrite [U] with the left singular vectors
			Zlacpy('U', &nr, n, a, lda, u, ldu)
			if nr > 1 {
				Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(1, 0), ldu)
			}
			//            .. the right singular vectors not computed, the NR left singular
			//            vectors overwrite [U](1:NR,1:NR)
			Zgesvd('O', 'N', &nr, n, u, ldu, s, u, ldu, v, ldv, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
			//               .. now [U](1:NR,1:NR) contains the NR left singular vectors of
			//               R. These will be pre-multiplied by Q to build the left singular
			//               vectors of A.
		}

		//           .. assemble the left singular vector matrix U of dimensions
		//              (M x NR) or (M x N) or (M x M).
		if (nr < (*m)) && (!wntuf) {
			Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
			if nr < n1 {
				Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
				Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
			}
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), &ierr)
		}
		if rowprm && !wntuf {
			Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, (*n)+1-1), toPtr(-1))
		}

	} else if rsvec && (!lsvec) {
		//.......................................................................
		//       .. the singular values and the right singular vectors requested
		//.......................................................................
		if rtrans {
			//            .. apply ZGESVD to R**H
			//            .. copy R**H into V and overwrite V with the left singular vectors
			for p = 1; p <= nr; p++ {
				for q = p; q <= (*n); q++ {
					v.Set(q-1, p-1, a.GetConj(p-1, q-1))
				}
			}
			if nr > 1 {
				Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
			}
			//           .. the left singular vectors of R**H overwrite V, the right singular
			//           vectors not computed
			if wntvr || (nr == (*n)) {
				Zgesvd('O', 'N', n, &nr, v, ldv, s, u, ldu, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)

				for p = 1; p <= nr; p++ {
					v.Set(p-1, p-1, v.GetConj(p-1, p-1))
					for q = p + 1; q <= nr; q++ {
						ctmp = v.GetConj(q-1, p-1)
						v.Set(q-1, p-1, v.GetConj(p-1, q-1))
						v.Set(p-1, q-1, ctmp)
					}
				}

				if nr < (*n) {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= (*n); q++ {
							v.Set(p-1, q-1, v.GetConj(q-1, p-1))
						}
					}
				}
				Zlapmt(false, &nr, n, v, ldv, iwork)
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:N,1:NR)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the QR factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Zlaset('G', n, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
				Zgesvd('O', 'N', n, n, v, ldv, s, u, ldu, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)

				for p = 1; p <= (*n); p++ {
					v.Set(p-1, p-1, v.GetConj(p-1, p-1))
					for q = p + 1; q <= (*n); q++ {
						ctmp = v.GetConj(q-1, p-1)
						v.Set(q-1, p-1, v.GetConj(p-1, q-1))
						v.Set(p-1, q-1, ctmp)
					}
				}
				Zlapmt(false, n, n, v, ldv, iwork)
			}

		} else {
			//            .. aply ZGESVD to R
			//            .. copy R into V and overwrite V with the right singular vectors
			Zlacpy('U', &nr, n, a, lda, v, ldv)
			if nr > 1 {
				Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(1, 0), ldv)
			}
			//            .. the right singular vectors overwrite V, the NR left singular
			//            vectors stored in U(1:NR,1:NR)
			if wntvr || (nr == (*n)) {
				Zgesvd('N', 'O', &nr, n, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
				Zlapmt(false, &nr, n, v, ldv, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**H
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:NR,1:N)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the LQ factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Zlaset('G', toPtr((*n)-nr), n, &czero, &czero, v.Off(nr+1-1, 0), ldv)
				Zgesvd('N', 'O', n, n, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
				Zlapmt(false, n, n, v, ldv, iwork)
			}
			//            .. now [V] contains the adjoint of the matrix of the right singular
			//            vectors of A.
		}

	} else {
		//.......................................................................
		//       .. FULL SVD requested
		//.......................................................................
		if rtrans {
			//            .. apply ZGESVD to R**H [[this option is left for R&D&T]]
			if wntvr || (nr == (*n)) {
				//            .. copy R**H into [V] and overwrite [V] with the left singular
				//            vectors of R**H
				for p = 1; p <= nr; p++ {
					for q = p; q <= (*n); q++ {
						v.Set(q-1, p-1, a.GetConj(p-1, q-1))
					}
				}
				if nr > 1 {
					Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
				}

				//           .. the left singular vectors of R**H overwrite [V], the NR right
				//           singular vectors of R**H stored in [U](1:NR,1:NR) as conjugate
				//           transposed
				Zgesvd('O', 'A', n, &nr, v, ldv, s, v, ldv, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
				//              .. assemble V
				for p = 1; p <= nr; p++ {
					v.Set(p-1, p-1, v.GetConj(p-1, p-1))
					for q = p + 1; q <= nr; q++ {
						ctmp = v.GetConj(q-1, p-1)
						v.Set(q-1, p-1, v.GetConj(p-1, q-1))
						v.Set(p-1, q-1, ctmp)
					}
				}
				if nr < (*n) {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= (*n); q++ {
							v.Set(p-1, q-1, v.GetConj(q-1, p-1))
						}
					}
				}
				Zlapmt(false, &nr, n, v, ldv, iwork)

				for p = 1; p <= nr; p++ {
					u.Set(p-1, p-1, u.GetConj(p-1, p-1))
					for q = p + 1; q <= nr; q++ {
						ctmp = u.GetConj(q-1, p-1)
						u.Set(q-1, p-1, u.GetConj(p-1, q-1))
						u.Set(p-1, q-1, ctmp)
					}
				}

				if (nr < (*m)) && !wntuf {
					Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
					if nr < n1 {
						Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
						Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
					}
				}

			} else {
				//               .. need all N right singular vectors and NR < N
				//            .. copy R**H into [V] and overwrite [V] with the left singular
				//            vectors of R**H
				//               [[The optimal ratio N/NR for using QRF instead of padding
				//                 with zeros. Here hard coded to 2; it must be at least
				//                 two due to work space constraints.]]
				//               OPTRATIO = ILAENV(6, 'ZGESVD', 'S' // 'O', NR,N,0,0)
				//               OPTRATIO = maxint( OPTRATIO, 2 )
				optratio = 2
				if optratio*nr > (*n) {
					for p = 1; p <= nr; p++ {
						for q = p; q <= (*n); q++ {
							v.Set(q-1, p-1, a.GetConj(p-1, q-1))
						}
					}
					if nr > 1 {
						Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
					}

					Zlaset('A', n, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
					Zgesvd('O', 'A', n, n, v, ldv, s, v, ldv, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)

					for p = 1; p <= (*n); p++ {
						v.Set(p-1, p-1, v.GetConj(p-1, p-1))
						for q = p + 1; q <= (*n); q++ {
							ctmp = v.GetConj(q-1, p-1)
							v.Set(q-1, p-1, v.GetConj(p-1, q-1))
							v.Set(p-1, q-1, ctmp)
						}
					}
					Zlapmt(false, n, n, v, ldv, iwork)
					//              .. assemble the left singular vector matrix U of dimensions
					//              (M x N1), i.e. (M x N) or (M x M).
					for p = 1; p <= (*n); p++ {
						u.Set(p-1, p-1, u.GetConj(p-1, p-1))
						for q = p + 1; q <= (*n); q++ {
							ctmp = u.GetConj(q-1, p-1)
							u.Set(q-1, p-1, u.GetConj(p-1, q-1))
							u.Set(p-1, q-1, ctmp)
						}
					}

					if ((*n) < (*m)) && !wntuf {
						Zlaset('A', toPtr((*m)-(*n)), n, &czero, &czero, u.Off((*n)+1-1, 0), ldu)
						if (*n) < n1 {
							Zlaset('A', n, toPtr(n1-(*n)), &czero, &czero, u.Off(0, (*n)+1-1), ldu)
							Zlaset('A', toPtr((*m)-(*n)), toPtr(n1-(*n)), &czero, &cone, u.Off((*n)+1-1, (*n)+1-1), ldu)
						}
					}
				} else {
					//                  .. copy R**H into [U] and overwrite [U] with the right
					//                  singular vectors of R
					for p = 1; p <= nr; p++ {
						for q = p; q <= (*n); q++ {
							u.Set(q-1, nr+p-1, a.GetConj(p-1, q-1))
						}
					}
					if nr > 1 {
						Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(0, nr+2-1), ldu)
					}
					Zgeqrf(n, &nr, u.Off(0, nr+1-1), ldu, cwork.Off((*n)+1-1), cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), &ierr)
					for p = 1; p <= nr; p++ {
						for q = 1; q <= (*n); q++ {
							v.Set(q-1, p-1, u.GetConj(p-1, nr+q-1))
						}
					}
					Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
					Zgesvd('S', 'O', &nr, &nr, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), rwork, info)
					Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
					Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
					Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
					Zunmqr('R', 'C', n, n, &nr, u.Off(0, nr+1-1), ldu, cwork.Off((*n)+1-1), v, ldv, cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), &ierr)
					Zlapmt(false, n, n, v, ldv, iwork)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x NR) or (M x N) or (M x M).
					if (nr < (*m)) && !wntuf {
						Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
						if nr < n1 {
							Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
							Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
						}
					}
				}
			}

		} else {
			//            .. apply ZGESVD to R [[this is the recommended option]]
			if wntvr || (nr == (*n)) {
				//                .. copy R into [V] and overwrite V with the right singular vectors
				Zlacpy('U', &nr, n, a, lda, v, ldv)
				if nr > 1 {
					Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(1, 0), ldv)
				}
				//               .. the right singular vectors of R overwrite [V], the NR left
				//               singular vectors of R stored in [U](1:NR,1:NR)
				Zgesvd('S', 'O', &nr, n, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
				Zlapmt(false, &nr, n, v, ldv, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**H
				//               .. assemble the left singular vector matrix U of dimensions
				//              (M x NR) or (M x N) or (M x M).
				if (nr < (*m)) && !wntuf {
					Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
					if nr < n1 {
						Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
						Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
					}
				}

			} else {
				//              .. need all N right singular vectors and NR < N
				//              .. the requested number of the left singular vectors
				//               is then N1 (N or M)
				//               [[The optimal ratio N/NR for using LQ instead of padding
				//                 with zeros. Here hard coded to 2; it must be at least
				//                 two due to work space constraints.]]
				//               OPTRATIO = ILAENV(6, 'ZGESVD', 'S' // 'O', NR,N,0,0)
				//               OPTRATIO = maxint( OPTRATIO, 2 )
				optratio = 2
				if optratio*nr > (*n) {
					Zlacpy('U', &nr, n, a, lda, v, ldv)
					if nr > 1 {
						Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(1, 0), ldv)
					}
					//              .. the right singular vectors of R overwrite [V], the NR left
					//                 singular vectors of R stored in [U](1:NR,1:NR)
					Zlaset('A', toPtr((*n)-nr), n, &czero, &czero, v.Off(nr+1-1, 0), ldv)
					Zgesvd('S', 'O', n, n, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), rwork, info)
					Zlapmt(false, n, n, v, ldv, iwork)
					//                 .. now [V] contains the adjoint of the matrix of the right
					//                 singular vectors of A. The leading N left singular vectors
					//                 are in [U](1:N,1:N)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x N1), i.e. (M x N) or (M x M).
					if ((*n) < (*m)) && !wntuf {
						Zlaset('A', toPtr((*m)-(*n)), n, &czero, &czero, u.Off((*n)+1-1, 0), ldu)
						if (*n) < n1 {
							Zlaset('A', n, toPtr(n1-(*n)), &czero, &czero, u.Off(0, (*n)+1-1), ldu)
							Zlaset('A', toPtr((*m)-(*n)), toPtr(n1-(*n)), &czero, &cone, u.Off((*n)+1-1, (*n)+1-1), ldu)
						}
					}
				} else {
					Zlacpy('U', &nr, n, a, lda, u.Off(nr+1-1, 0), ldu)
					if nr > 1 {
						Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(nr+2-1, 0), ldu)
					}
					Zgelqf(&nr, n, u.Off(nr+1-1, 0), ldu, cwork.Off((*n)+1-1), cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), &ierr)
					Zlacpy('L', &nr, &nr, u.Off(nr+1-1, 0), ldu, v, ldv)
					if nr > 1 {
						Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
					}
					Zgesvd('S', 'O', &nr, &nr, v, ldv, s, u, ldu, v, ldv, cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), rwork, info)
					Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
					Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
					Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
					Zunmlq('R', 'N', n, n, &nr, u.Off(nr+1-1, 0), ldu, cwork.Off((*n)+1-1), v, ldv, cwork.Off((*n)+nr+1-1), toPtr((*lcwork)-(*n)-nr), &ierr)
					Zlapmt(false, n, n, v, ldv, iwork)
					//               .. assemble the left singular vector matrix U of dimensions
					//              (M x NR) or (M x N) or (M x M).
					if (nr < (*m)) && !wntuf {
						Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
						if nr < n1 {
							Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
							Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
						}
					}
				}
			}
			//        .. end of the "R**H or R" branch
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lcwork)-(*n)), &ierr)
		}
		if rowprm && !wntuf {
			Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, (*n)+1-1), toPtr(-1))
		}

		//     ... end of the "full SVD" branch
	}

	//     Check whether some singular values are returned as zeros, e.g.
	//     due to underflow, and update the numerical rank.
	p = nr
	for q = p; q >= 1; q -= 1 {
		if s.Get(q-1) > zero {
			goto label4002
		}
		nr = nr - 1
	}
label4002:
	;

	//     .. if numerical rank deficiency is detected, the truncated
	//     singular values are set to zero.
	if nr < (*n) {
		Dlaset('G', toPtr((*n)-nr), func() *int { y := 1; return &y }(), &zero, &zero, s.MatrixOff(nr+1-1, *n, opts), n)
	}
	//     .. undo scaling; this may cause overflow in the largest singular
	//     values.
	if ascaled {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &one, toPtrf64(math.Sqrt(float64(*m))), &nr, func() *int { y := 1; return &y }(), s.Matrix(*n, opts), n, &ierr)
	}
	if conda {
		rwork.Set(0, sconda)
	}
	rwork.Set(1, float64(p-nr))
	//     .. p-NR is the number of singular values that are computed as
	//     exact zeros in ZGESVD() applied to the (possibly truncated)
	//     full row rank triangular (trapezoidal) factor of A.
	(*numrank) = nr
}
