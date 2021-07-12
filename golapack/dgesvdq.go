package golapack

import (
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dgesvdq computes the singular value decomposition (SVD) of a real
// M-by-N matrix A, where M >= N. The SVD of A is written as
//                                    [++]   [xx]   [x0]   [xx]
//              A = U * SIGMA * V^*,  [++] = [xx] * [ox] * [xx]
//                                    [++]   [xx]
// where SIGMA is an N-by-N diagonal matrix, U is an M-by-N orthonormal
// matrix, and V is an N-by-N orthogonal matrix. The diagonal elements
// of SIGMA are the singular values of A. The columns of U and V are the
// left and the right singular vectors of A, respectively.
func Dgesvdq(joba, jobp, jobr, jobu, jobv byte, m, n *int, a *mat.Matrix, lda *int, s *mat.Vector, u *mat.Matrix, ldu *int, v *mat.Matrix, ldv *int, numrank *int, iwork *[]int, liwork *int, work *mat.Vector, lwork *int, rwork *mat.Vector, lrwork, info *int) {
	var accla, acclh, acclm, ascaled, conda, dntwu, dntwv, lquery, lsvc0, lsvec, rowprm, rsvec, rtrans, wntua, wntuf, wntur, wntus, wntva, wntvr bool
	var big, epsln, one, rtmp, sconda, sfmin, zero float64
	var ierr, iminwrk, iwoff, lwcon, lwlqf, lworlq, lworq, lworq2, lwqp3, lwqrf, lwrkDgelqf, lwrkDgeqp3, lwrkDgeqrf, lwrkDgesvd, lwrkDgesvd2, lwrkDormlq, lwrkDormqr, lwrkDormqr2, lwsvd, lwsvd2, minwrk, minwrk2, n1, nr, optratio, optwrk, optwrk2, p, q, rminwrk int

	rdummy := vf(1)

	zero = 0.0
	one = 1.0

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
		if conda {
			iminwrk = max(1, (*n)+(*m)-1+(*n))
		} else {
			iminwrk = max(1, (*n)+(*m)-1)
		}
		rminwrk = max(2, *m)
	} else {
		if conda {
			iminwrk = max(1, (*n)+(*n))
		} else {
			iminwrk = max(1, *n)
		}
		rminwrk = 2
	}
	lquery = ((*liwork) == -1 || (*lwork) == -1 || (*lrwork) == -1)
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
	} else if (*lda) < max(1, *m) {
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
		//        values of LWORK are written with a lot of redundancy and
		//        can be simplified. However, this detailed form is easier for
		//        maintenance and modifications of the code.]]
		//
		//        .. minimal workspace length for DGEQP3 of an M x N matrix
		lwqp3 = 3*(*n) + 1
		//        .. minimal workspace length for DORMQR to build left singular vectors
		if wntus || wntur {
			lworq = max(*n, 1)
		} else if wntua {
			lworq = max(*m, 1)
		}
		//        .. minimal workspace length for DPOCON of an N x N matrix
		lwcon = 3 * (*n)
		//        .. DGESVD of an N x N matrix
		lwsvd = max(5*(*n), 1)
		if lquery {
			Dgeqp3(m, n, a, lda, iwork, rdummy, rdummy, toPtr(-1), &ierr)
			lwrkDgeqp3 = int(rdummy.Get(0))
			if wntus || wntur {
				Dormqr('L', 'N', m, n, n, a, lda, rdummy, u, ldu, rdummy, toPtr(-1), &ierr)
				lwrkDormqr = int(rdummy.Get(0))
			} else if wntua {
				Dormqr('L', 'N', m, m, n, a, lda, rdummy, u, ldu, rdummy, toPtr(-1), &ierr)
				lwrkDormqr = int(rdummy.Get(0))
			} else {
				lwrkDormqr = 0
			}
		}
		minwrk = 2
		optwrk = 2
		if !(lsvec || rsvec) {
			//            .. minimal and optimal sizes of the workspace if
			//            only the singular values are requested
			if conda {
				minwrk = max((*n)+lwqp3, lwcon, lwsvd)
			} else {
				minwrk = max((*n)+lwqp3, lwsvd)
			}
			if lquery {
				Dgesvd('N', 'N', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = max((*n)+lwrkDgeqp3, (*n)+lwcon, lwrkDgesvd)
				} else {
					optwrk = max((*n)+lwrkDgeqp3, lwrkDgesvd)
				}
			}
		} else if lsvec && (!rsvec) {
			//            .. minimal and optimal sizes of the workspace if the
			//            singular values and the left singular vectors are requested
			if conda {
				minwrk = (*n) + max(lwqp3, lwcon, lwsvd, lworq)
			} else {
				minwrk = (*n) + max(lwqp3, lwsvd, lworq)
			}
			if lquery {
				if rtrans {
					Dgesvd('N', 'O', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
				} else {
					Dgesvd('O', 'N', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
				}
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = (*n) + max(lwrkDgeqp3, lwcon, lwrkDgesvd, lwrkDormqr)
				} else {
					optwrk = (*n) + max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
				}
			}
		} else if rsvec && (!lsvec) {
			//            .. minimal and optimal sizes of the workspace if the
			//            singular values and the right singular vectors are requested
			if conda {
				minwrk = (*n) + max(lwqp3, lwcon, lwsvd)
			} else {
				minwrk = (*n) + max(lwqp3, lwsvd)
			}
			if lquery {
				if rtrans {
					Dgesvd('O', 'N', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
				} else {
					Dgesvd('N', 'O', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
				}
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = (*n) + max(lwrkDgeqp3, lwcon, lwrkDgesvd)
				} else {
					optwrk = (*n) + max(lwrkDgeqp3, lwrkDgesvd)
				}
			}
		} else {
			//            .. minimal and optimal sizes of the workspace if the
			//            full SVD is requested
			if rtrans {
				minwrk = max(lwqp3, lwsvd, lworq)
				if conda {
					minwrk = max(minwrk, lwcon)
				}
				minwrk = minwrk + (*n)
				if wntva {
					//                   .. minimal workspace length for N x N/2 DGEQRF
					lwqrf = max((*n)/2, 1)
					//                   .. minimal workspace lengt for N/2 x N/2 DGESVD
					lwsvd2 = max(5*((*n)/2), 1)
					lworq2 = max(*n, 1)
					minwrk2 = max(lwqp3, (*n)/2+lwqrf, (*n)/2+lwsvd2, (*n)/2+lworq2, lworq)
					if conda {
						minwrk2 = max(minwrk2, lwcon)
					}
					minwrk2 = (*n) + minwrk2
					minwrk = max(minwrk, minwrk2)
				}
			} else {
				minwrk = max(lwqp3, lwsvd, lworq)
				if conda {
					minwrk = max(minwrk, lwcon)
				}
				minwrk = minwrk + (*n)
				if wntva {
					//                   .. minimal workspace length for N/2 x N DGELQF
					lwlqf = max((*n)/2, 1)
					lwsvd2 = max(5*((*n)/2), 1)
					lworlq = max(*n, 1)
					minwrk2 = max(lwqp3, (*n)/2+lwlqf, (*n)/2+lwsvd2, (*n)/2+lworlq, lworq)
					if conda {
						minwrk2 = max(minwrk2, lwcon)
					}
					minwrk2 = (*n) + minwrk2
					minwrk = max(minwrk, minwrk2)
				}
			}
			if lquery {
				if rtrans {
					Dgesvd('O', 'A', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
					lwrkDgesvd = int(rdummy.Get(0))
					optwrk = max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
					if conda {
						optwrk = max(optwrk, lwcon)
					}
					optwrk = (*n) + optwrk
					if wntva {
						Dgeqrf(n, toPtr((*n)/2), u, ldu, rdummy, rdummy, toPtr(-1), &ierr)
						lwrkDgeqrf = int(rdummy.Get(0))
						Dgesvd('S', 'O', toPtr((*n)/2), toPtr((*n)/2), v, ldv, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
						lwrkDgesvd2 = int(rdummy.Get(0))
						Dormqr('R', 'C', n, n, toPtr((*n)/2), u, ldu, rdummy, v, ldv, rdummy, toPtr(-1), &ierr)
						lwrkDormqr2 = int(rdummy.Get(0))
						optwrk2 = max(lwrkDgeqp3, (*n)/2+lwrkDgeqrf, (*n)/2+lwrkDgesvd2, (*n)/2+lwrkDormqr2)
						if conda {
							optwrk2 = max(optwrk2, lwcon)
						}
						optwrk2 = (*n) + optwrk2
						optwrk = max(optwrk, optwrk2)
					}
				} else {
					Dgesvd('S', 'O', n, n, a, lda, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
					lwrkDgesvd = int(rdummy.Get(0))
					optwrk = max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
					if conda {
						optwrk = max(optwrk, lwcon)
					}
					optwrk = (*n) + optwrk
					if wntva {
						Dgelqf(toPtr((*n)/2), n, u, ldu, rdummy, rdummy, toPtr(-1), &ierr)
						lwrkDgelqf = int(rdummy.Get(0))
						Dgesvd('S', 'O', toPtr((*n)/2), toPtr((*n)/2), v, ldv, s, u, ldu, v, ldv, rdummy, toPtr(-1), &ierr)
						lwrkDgesvd2 = int(rdummy.Get(0))
						Dormlq('R', 'N', n, n, toPtr((*n)/2), u, ldu, rdummy, v, ldv, rdummy, toPtr(-1), &ierr)
						lwrkDormlq = int(rdummy.Get(0))
						optwrk2 = max(lwrkDgeqp3, (*n)/2+lwrkDgelqf, (*n)/2+lwrkDgesvd2, (*n)/2+lwrkDormlq)
						if conda {
							optwrk2 = max(optwrk2, lwcon)
						}
						optwrk2 = (*n) + optwrk2
						optwrk = max(optwrk, optwrk2)
					}
				}
			}
		}

		minwrk = max(2, minwrk)
		optwrk = max(2, optwrk)
		if (*lwork) < minwrk && (!lquery) {
			(*info) = -19
		}

	}

	if (*info) == 0 && (*lrwork) < rminwrk && !lquery {
		(*info) = -21
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DGESVDQ"), -(*info))
		return
	} else if lquery {
		//     Return optimal workspace
		(*iwork)[0] = iminwrk
		work.Set(0, float64(optwrk))
		work.Set(1, float64(minwrk))
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
	iwoff = 1
	if rowprm {
		iwoff = (*m)
		//           .. reordering the rows in decreasing sequence in the
		//           ell-infinity norm - this enhances numerical robustness in
		//           the case of differently scaled rows.
		for p = 1; p <= (*m); p++ {
			//               RWORK(p) = ABS( A(p,ICAMAX(N,A(p,1),LDA)) )
			//               [[DLANGE will return NaN if an entry of the p-th row is Nan]]
			rwork.Set(p-1, Dlange('M', toPtr(1), n, a.Off(p-1, 0), lda, rdummy))
			//               .. check for NaN's and Inf's
			if (rwork.Get(p-1) != rwork.Get(p-1)) || ((rwork.Get(p-1) * zero) != zero) {
				(*info) = -8
				gltest.Xerbla([]byte("DGESVDQ"), -(*info))
				return
			}
		}
		for p = 1; p <= (*m)-1; p++ {
			q = goblas.Idamax((*m)-p+1, rwork.Off(p-1)) + p - 1
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
			Dlaset('G', n, toPtr(1), &zero, &zero, s.Matrix(*n, opts), n)
			if wntus {
				Dlaset('G', m, n, &zero, &one, u, ldu)
			}
			if wntua {
				Dlaset('G', m, m, &zero, &one, u, ldu)
			}
			if wntva {
				Dlaset('G', n, n, &zero, &one, v, ldv)
			}
			if wntuf {
				Dlaset('G', n, toPtr(1), &zero, &zero, work.Matrix(*n, opts), n)
				Dlaset('G', m, n, &zero, &one, u, ldu)
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
			//               matrix by 1/math.Sqrt(M) if too large entry detected
			Dlascl('G', toPtr(0), toPtr(0), toPtrf64(math.Sqrt(float64(*m))), &one, m, n, a, lda, &ierr)
			ascaled = true
		}
		Dlaswp(n, a, lda, toPtr(1), toPtr((*m)-1), toSlice(iwork, (*n)), toPtr(1))
	}

	//    .. At this stage, preemptive scaling is done only to avoid column
	//    norms overflows during the QR factorization. The SVD procedure should
	//    have its own scaling to save the singular values from overflows and
	//    underflows. That depends on the SVD procedure.
	if !rowprm {
		rtmp = Dlange('M', m, n, a, lda, rdummy)
		if (rtmp != rtmp) || ((rtmp * zero) != zero) {
			(*info) = -8
			gltest.Xerbla([]byte("DGESVDQ"), -(*info))
			return
		}
		if rtmp > big/math.Sqrt(float64(*m)) {
			//             .. to prevent overflow in the QR factorization, scale the
			//             matrix by 1/math.Sqrt(M) if too large entry detected
			Dlascl('G', toPtr(0), toPtr(0), toPtrf64(math.Sqrt(float64(*m))), &one, m, n, a, lda, &ierr)
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
	Dgeqp3(m, n, a, lda, iwork, work, work.Off((*n)), toPtr((*lwork)-(*n)), &ierr)

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
			if math.Abs(a.Get(p-1, p-1)) < (rtmp * math.Abs(a.Get(0, 0))) {
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
			if (math.Abs(a.Get(p-1, p-1)) < (epsln * math.Abs(a.Get(p-1-1, p-1-1)))) || (math.Abs(a.Get(p-1, p-1)) < sfmin) {
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
			if math.Abs(a.Get(p-1, p-1)) == zero {
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
			Dlacpy('U', n, n, a, lda, v, ldv)
			//              Only the leading NR x NR submatrix of the triangular factor
			//              is considered. Only if NR=N will this give a reliable error
			//              bound. However, even for NR < N, this can be used on an
			//              expert level and obtain useful information in the sense of
			//              perturbation theory.
			for p = 1; p <= nr; p++ {
				rtmp = goblas.Dnrm2(p, v.Vector(0, p-1, 1))
				goblas.Dscal(p, one/rtmp, v.Vector(0, p-1, 1))
			}
			if !(lsvec || rsvec) {
				Dpocon('U', &nr, v, ldv, &one, &rtmp, work, toSlice(iwork, (*n)+iwoff-1), &ierr)
			} else {
				Dpocon('U', &nr, v, ldv, &one, &rtmp, work.Off((*n)), toSlice(iwork, (*n)+iwoff-1), &ierr)
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
			//         .. compute the singular values of R**T = [A](1:NR,1:N)**T
			//           .. set the lower triangle of [A] to [A](1:NR,1:N)**T and
			//           the upper triangle of [A] to zero.
			for p = 1; p <= min(*n, nr); p++ {
				for q = p + 1; q <= (*n); q++ {
					a.Set(q-1, p-1, a.Get(p-1, q-1))
					if q <= nr {
						a.Set(p-1, q-1, zero)
					}
				}
			}

			Dgesvd('N', 'N', n, &nr, a, lda, s, u, ldu, v, ldv, work, lwork, info)

		} else {
			//           .. compute the singular values of R = [A](1:NR,1:N)
			if nr > 1 {
				Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, a.Off(1, 0), lda)
			}
			Dgesvd('N', 'N', &nr, n, a, lda, s, u, ldu, v, ldv, work, lwork, info)

		}

	} else if lsvec && (!rsvec) {
		//.......................................................................
		//       .. the singular values and the left singular vectors requested
		//.......................................................................""""""""
		if rtrans {
			//            .. apply DGESVD to R**T
			//            .. copy R**T into [U] and overwrite [U] with the right singular
			//            vectors of R
			for p = 1; p <= nr; p++ {
				for q = p; q <= (*n); q++ {
					u.Set(q-1, p-1, a.Get(p-1, q-1))
				}
			}
			if nr > 1 {
				Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, u.Off(0, 1), ldu)
			}
			//           .. the left singular vectors not computed, the NR right singular
			//           vectors overwrite [U](1:NR,1:NR) as transposed. These
			//           will be pre-multiplied by Q to build the left singular vectors of A.
			Dgesvd('N', 'O', n, &nr, u, ldu, s, u, ldu, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), info)

			for p = 1; p <= nr; p++ {
				for q = p + 1; q <= nr; q++ {
					rtmp = u.Get(q-1, p-1)
					u.Set(q-1, p-1, u.Get(p-1, q-1))
					u.Set(p-1, q-1, rtmp)
				}
			}

		} else {
			//            .. apply DGESVD to R
			//            .. copy R into [U] and overwrite [U] with the left singular vectors
			Dlacpy('U', &nr, n, a, lda, u, ldu)
			if nr > 1 {
				Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, u.Off(1, 0), ldu)
			}
			//            .. the right singular vectors not computed, the NR left singular
			//            vectors overwrite [U](1:NR,1:NR)
			Dgesvd('O', 'N', &nr, n, u, ldu, s, u, ldu, v, ldv, work.Off((*n)), toPtr((*lwork)-(*n)), info)
			//               .. now [U](1:NR,1:NR) contains the NR left singular vectors of
			//               R. These will be pre-multiplied by Q to build the left singular
			//               vectors of A.
		}

		//           .. assemble the left singular vector matrix U of dimensions
		//              (M x NR) or (M x N) or (M x M).
		if (nr < (*m)) && (!wntuf) {
			Dlaset('A', toPtr((*m)-nr), &nr, &zero, &zero, u.Off(nr, 0), ldu)
			if nr < n1 {
				Dlaset('A', &nr, toPtr(n1-nr), &zero, &zero, u.Off(0, nr), ldu)
				Dlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &zero, &one, u.Off(nr, nr), ldu)
			}
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			Dormqr('L', 'N', m, &n1, n, a, lda, work, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), &ierr)
		}
		if rowprm && !wntuf {
			Dlaswp(&n1, u, ldu, toPtr(1), toPtr((*m)-1), toSlice(iwork, (*n)), toPtr(-1))
		}

	} else if rsvec && (!lsvec) {
		//.......................................................................
		//       .. the singular values and the right singular vectors requested
		//.......................................................................
		if rtrans {
			//            .. apply DGESVD to R**T
			//            .. copy R**T into V and overwrite V with the left singular vectors
			for p = 1; p <= nr; p++ {
				for q = p; q <= (*n); q++ {
					v.Set(q-1, p-1, (a.Get(p-1, q-1)))
				}
			}
			if nr > 1 {
				Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(0, 1), ldv)
			}
			//           .. the left singular vectors of R**T overwrite V, the right singular
			//           vectors not computed
			if wntvr || (nr == (*n)) {
				Dgesvd('O', 'N', n, &nr, v, ldv, s, u, ldu, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), info)

				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}

				if nr < (*n) {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= (*n); q++ {
							v.Set(p-1, q-1, v.Get(q-1, p-1))
						}
					}
				}
				Dlapmt(false, &nr, n, v, ldv, iwork)
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:N,1:NR)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the QR factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Dlaset('G', n, toPtr((*n)-nr), &zero, &zero, v.Off(0, nr), ldv)
				Dgesvd('O', 'N', n, n, v, ldv, s, u, ldu, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), info)

				for p = 1; p <= (*n); p++ {
					for q = p + 1; q <= (*n); q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}
				Dlapmt(false, n, n, v, ldv, iwork)
			}

		} else {
			//            .. aply DGESVD to R
			//            .. copy R into V and overwrite V with the right singular vectors
			Dlacpy('U', &nr, n, a, lda, v, ldv)
			if nr > 1 {
				Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(1, 0), ldv)
			}
			//            .. the right singular vectors overwrite V, the NR left singular
			//            vectors stored in U(1:NR,1:NR)
			if wntvr || (nr == (*n)) {
				Dgesvd('N', 'O', &nr, n, v, ldv, s, u, ldu, v, ldv, work.Off((*n)), toPtr((*lwork)-(*n)), info)
				Dlapmt(false, &nr, n, v, ldv, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**T
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:NR,1:N)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the LQ factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Dlaset('G', toPtr((*n)-nr), n, &zero, &zero, v.Off(nr, 0), ldv)
				Dgesvd('N', 'O', n, n, v, ldv, s, u, ldu, v, ldv, work.Off((*n)), toPtr((*lwork)-(*n)), info)
				Dlapmt(false, n, n, v, ldv, iwork)
			}
			//            .. now [V] contains the transposed matrix of the right singular
			//            vectors of A.
		}

	} else {
		//.......................................................................
		//       .. FULL SVD requested
		//.......................................................................
		if rtrans {
			//            .. apply DGESVD to R**T [[this option is left for R&D&T]]
			if wntvr || (nr == (*n)) {
				//            .. copy R**T into [V] and overwrite [V] with the left singular
				//            vectors of R**T
				for p = 1; p <= nr; p++ {
					for q = p; q <= (*n); q++ {
						v.Set(q-1, p-1, a.Get(p-1, q-1))
					}
				}
				if nr > 1 {
					Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(0, 1), ldv)
				}

				//           .. the left singular vectors of R**T overwrite [V], the NR right
				//           singular vectors of R**T stored in [U](1:NR,1:NR) as transposed
				Dgesvd('O', 'A', n, &nr, v, ldv, s, v, ldv, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), info)
				//              .. assemble V
				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}
				if nr < (*n) {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= (*n); q++ {
							v.Set(p-1, q-1, v.Get(q-1, p-1))
						}
					}
				}
				Dlapmt(false, &nr, n, v, ldv, iwork)

				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = u.Get(q-1, p-1)
						u.Set(q-1, p-1, u.Get(p-1, q-1))
						u.Set(p-1, q-1, rtmp)
					}
				}

				if (nr < (*m)) && !wntuf {
					Dlaset('A', toPtr((*m)-nr), &nr, &zero, &zero, u.Off(nr, 0), ldu)
					if nr < n1 {
						Dlaset('A', &nr, toPtr(n1-nr), &zero, &zero, u.Off(0, nr), ldu)
						Dlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &zero, &one, u.Off(nr, nr), ldu)
					}
				}

			} else {
				//               .. need all N right singular vectors and NR < N
				//            .. copy R**T into [V] and overwrite [V] with the left singular
				//            vectors of R**T
				//               [[The optimal ratio N/NR for using QRF instead of padding
				//                 with zeros. Here hard coded to 2; it must be at least
				//                 two due to work space constraints.]]
				//               OPTRATIO = ILAENV(6, 'DGESVD', 'S' // 'O', NR,N,0,0)
				//               OPTRATIO = max( OPTRATIO, 2 )
				optratio = 2
				if optratio*nr > (*n) {
					for p = 1; p <= nr; p++ {
						for q = p; q <= (*n); q++ {
							v.Set(q-1, p-1, a.Get(p-1, q-1))
						}
					}
					if nr > 1 {
						Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(0, 1), ldv)
					}

					Dlaset('A', n, toPtr((*n)-nr), &zero, &zero, v.Off(0, nr), ldv)
					Dgesvd('O', 'A', n, n, v, ldv, s, v, ldv, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), info)

					for p = 1; p <= (*n); p++ {
						for q = p + 1; q <= (*n); q++ {
							rtmp = v.Get(q-1, p-1)
							v.Set(q-1, p-1, v.Get(p-1, q-1))
							v.Set(p-1, q-1, rtmp)
						}
					}
					Dlapmt(false, n, n, v, ldv, iwork)
					//              .. assemble the left singular vector matrix U of dimensions
					//              (M x N1), i.e. (M x N) or (M x M).

					for p = 1; p <= (*n); p++ {
						for q = p + 1; q <= (*n); q++ {
							rtmp = u.Get(q-1, p-1)
							u.Set(q-1, p-1, u.Get(p-1, q-1))
							u.Set(p-1, q-1, rtmp)
						}
					}

					if ((*n) < (*m)) && !wntuf {
						Dlaset('A', toPtr((*m)-(*n)), n, &zero, &zero, u.Off((*n), 0), ldu)
						if (*n) < n1 {
							Dlaset('A', n, toPtr(n1-(*n)), &zero, &zero, u.Off(0, (*n)), ldu)
							Dlaset('A', toPtr((*m)-(*n)), toPtr(n1-(*n)), &zero, &one, u.Off((*n), (*n)), ldu)
						}
					}
				} else {
					//                  .. copy R**T into [U] and overwrite [U] with the right
					//                  singular vectors of R
					for p = 1; p <= nr; p++ {
						for q = p; q <= (*n); q++ {
							u.Set(q-1, nr+p-1, a.Get(p-1, q-1))
						}
					}
					if nr > 1 {
						Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, u.Off(0, nr+2-1), ldu)
					}
					Dgeqrf(n, &nr, u.Off(0, nr), ldu, work.Off((*n)), work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), &ierr)
					for p = 1; p <= nr; p++ {
						for q = 1; q <= (*n); q++ {
							v.Set(q-1, p-1, u.Get(p-1, nr+q-1))
						}
					}
					Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(0, 1), ldv)
					Dgesvd('S', 'O', &nr, &nr, v, ldv, s, u, ldu, v, ldv, work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), info)
					Dlaset('A', toPtr((*n)-nr), &nr, &zero, &zero, v.Off(nr, 0), ldv)
					Dlaset('A', &nr, toPtr((*n)-nr), &zero, &zero, v.Off(0, nr), ldv)
					Dlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &zero, &one, v.Off(nr, nr), ldv)
					Dormqr('R', 'C', n, n, &nr, u.Off(0, nr), ldu, work.Off((*n)), v, ldv, work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), &ierr)
					Dlapmt(false, n, n, v, ldv, iwork)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x NR) or (M x N) or (M x M).
					if (nr < (*m)) && !wntuf {
						Dlaset('A', toPtr((*m)-nr), &nr, &zero, &zero, u.Off(nr, 0), ldu)
						if nr < n1 {
							Dlaset('A', &nr, toPtr(n1-nr), &zero, &zero, u.Off(0, nr), ldu)
							Dlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &zero, &one, u.Off(nr, nr), ldu)
						}
					}
				}
			}

		} else {
			//            .. apply DGESVD to R [[this is the recommended option]]
			if wntvr || (nr == (*n)) {
				//                .. copy R into [V] and overwrite V with the right singular vectors
				Dlacpy('U', &nr, n, a, lda, v, ldv)
				if nr > 1 {
					Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(1, 0), ldv)
				}
				//               .. the right singular vectors of R overwrite [V], the NR left
				//               singular vectors of R stored in [U](1:NR,1:NR)
				Dgesvd('S', 'O', &nr, n, v, ldv, s, u, ldu, v, ldv, work.Off((*n)), toPtr((*lwork)-(*n)), info)
				Dlapmt(false, &nr, n, v, ldv, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**T
				//               .. assemble the left singular vector matrix U of dimensions
				//              (M x NR) or (M x N) or (M x M).
				if (nr < (*m)) && !wntuf {
					Dlaset('A', toPtr((*m)-nr), &nr, &zero, &zero, u.Off(nr, 0), ldu)
					if nr < n1 {
						Dlaset('A', &nr, toPtr(n1-nr), &zero, &zero, u.Off(0, nr), ldu)
						Dlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &zero, &one, u.Off(nr, nr), ldu)
					}
				}

			} else {
				//              .. need all N right singular vectors and NR < N
				//              .. the requested number of the left singular vectors
				//               is then N1 (N or M)
				//               [[The optimal ratio N/NR for using LQ instead of padding
				//                 with zeros. Here hard coded to 2; it must be at least
				//                 two due to work space constraints.]]
				//               OPTRATIO = ILAENV(6, 'DGESVD', 'S' // 'O', NR,N,0,0)
				//               OPTRATIO = max( OPTRATIO, 2 )
				optratio = 2
				if optratio*nr > (*n) {
					Dlacpy('U', &nr, n, a, lda, v, ldv)
					if nr > 1 {
						Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(1, 0), ldv)
					}
					//              .. the right singular vectors of R overwrite [V], the NR left
					//                 singular vectors of R stored in [U](1:NR,1:NR)
					Dlaset('A', toPtr((*n)-nr), n, &zero, &zero, v.Off(nr, 0), ldv)
					Dgesvd('S', 'O', n, n, v, ldv, s, u, ldu, v, ldv, work.Off((*n)), toPtr((*lwork)-(*n)), info)
					Dlapmt(false, n, n, v, ldv, iwork)
					//                 .. now [V] contains the transposed matrix of the right
					//                 singular vectors of A. The leading N left singular vectors
					//                 are in [U](1:N,1:N)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x N1), i.e. (M x N) or (M x M).
					if ((*n) < (*m)) && !wntuf {
						Dlaset('A', toPtr((*m)-(*n)), n, &zero, &zero, u.Off((*n), 0), ldu)
						if (*n) < n1 {
							Dlaset('A', n, toPtr(n1-(*n)), &zero, &zero, u.Off(0, (*n)), ldu)
							Dlaset('A', toPtr((*m)-(*n)), toPtr(n1-(*n)), &zero, &one, u.Off((*n), (*n)), ldu)
						}
					}
				} else {
					Dlacpy('U', &nr, n, a, lda, u.Off(nr, 0), ldu)
					if nr > 1 {
						Dlaset('L', toPtr(nr-1), toPtr(nr-1), &zero, &zero, u.Off(nr+2-1, 0), ldu)
					}
					Dgelqf(&nr, n, u.Off(nr, 0), ldu, work.Off((*n)), work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), &ierr)
					Dlacpy('L', &nr, &nr, u.Off(nr, 0), ldu, v, ldv)
					if nr > 1 {
						Dlaset('U', toPtr(nr-1), toPtr(nr-1), &zero, &zero, v.Off(0, 1), ldv)
					}
					Dgesvd('S', 'O', &nr, &nr, v, ldv, s, u, ldu, v, ldv, work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), info)
					Dlaset('A', toPtr((*n)-nr), &nr, &zero, &zero, v.Off(nr, 0), ldv)
					Dlaset('A', &nr, toPtr((*n)-nr), &zero, &zero, v.Off(0, nr), ldv)
					Dlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &zero, &one, v.Off(nr, nr), ldv)
					Dormlq('R', 'N', n, n, &nr, u.Off(nr, 0), ldu, work.Off((*n)), v, ldv, work.Off((*n)+nr), toPtr((*lwork)-(*n)-nr), &ierr)
					Dlapmt(false, n, n, v, ldv, iwork)
					//               .. assemble the left singular vector matrix U of dimensions
					//              (M x NR) or (M x N) or (M x M).
					if (nr < (*m)) && !wntuf {
						Dlaset('A', toPtr((*m)-nr), &nr, &zero, &zero, u.Off(nr, 0), ldu)
						if nr < n1 {
							Dlaset('A', &nr, toPtr(n1-nr), &zero, &zero, u.Off(0, nr), ldu)
							Dlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &zero, &one, u.Off(nr, nr), ldu)
						}
					}
				}
			}
			//        .. end of the "R**T or R" branch
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			Dormqr('L', 'N', m, &n1, n, a, lda, work, u, ldu, work.Off((*n)), toPtr((*lwork)-(*n)), &ierr)
		}
		if rowprm && !wntuf {
			Dlaswp(&n1, u, ldu, toPtr(1), toPtr((*m)-1), toSlice(iwork, (*n)), toPtr(-1))
		}

		//     ... end of the "full SVD" branch
	}

	//     Check whether some singular values are returned as zeros, e.g.
	//     due to underflow, and update the numerical rank.
	p = nr
	for q = p; q >= 1; q-- {
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
		Dlaset('G', toPtr((*n)-nr), toPtr(1), &zero, &zero, s.MatrixOff(nr, *n, opts), n)
	}
	//     .. undo scaling; this may cause overflow in the largest singular
	//     values.
	if ascaled {
		Dlascl('G', toPtr(0), toPtr(0), &one, toPtrf64(math.Sqrt(float64(*m))), &nr, toPtr(1), s.Matrix(*n, opts), n, &ierr)
	}
	if conda {
		rwork.Set(0, sconda)
	}
	rwork.Set(1, float64(p-nr))
	//     .. p-NR is the number of singular values that are computed as
	//     exact zeros in DGESVD() applied to the (possibly truncated)
	//     full row rank triangular (trapezoidal) factor of A.
	(*numrank) = nr
}
