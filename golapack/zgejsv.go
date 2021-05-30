package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
	"math/cmplx"
)

// Zgejsv computes the singular value decomposition (SVD) of a complex M-by-N
// matrix [A], where M >= N. The SVD of [A] is written as
//
//              [A] = [U] * [SIGMA] * [V]^*,
//
// where [SIGMA] is an N-by-N (M-by-N) matrix which is zero except for its N
// diagonal elements, [U] is an M-by-N (or M-by-M) unitary matrix, and
// [V] is an N-by-N unitary matrix. The diagonal elements of [SIGMA] are
// the singular values of [A]. The columns of [U] and [V] are the left and
// the right singular vectors of [A], respectively. The matrices [U] and [V]
// are computed and stored in the arrays U and V, respectively. The diagonal
// of [SIGMA] is computed and stored in the array SVA.
func Zgejsv(joba, jobu, jobv, jobr, jobt, jobp byte, m, n *int, a *mat.CMatrix, lda *int, sva *mat.Vector, u *mat.CMatrix, ldu *int, v *mat.CMatrix, ldv *int, cwork *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, info *int) {
	var almort, defr, errest, goscal, jracc, kill, l2aber, l2kill, l2pert, l2rank, l2tran, lquery, lsvec, noscal, rowpiv, rsvec, transp bool
	var cone, ctemp, czero complex128
	var aapp, aaqq, aatmax, aatmin, big, big1, condOk, condr1, condr2, entra, entrat, epsln, maxprj, one, scalem, sconda, sfmin, small, temp1, uscal1, uscal2, xsc, zero float64
	var ierr, iwoff, lrwcon, lrwqp3, lrwsvdj, lwcon, lwlqf, lwqp3, lwqrf, lwrkZgelqf, lwrkZgeqp3, lwrkZgeqp3n, lwrkZgeqrf, lwrkZgesvj, lwrkZgesvju, lwrkZgesvjv, lwrkZunmlq, lwrkZunmqr, lwrkZunmqrm, lwsvdj, lwsvdjv, lwunmlq, lwunmqr, lwunmqrm, miniwrk, minrwrk, minwrk, n1, nr, numrank, optwrk, p, q, warning int
	cdummy := cvf(1)
	rdummy := vf(1)

	zero = 0.0
	one = 1.0
	czero = (0.0 + 0.0*1i)
	cone = (1.0 + 0.0*1i)

	//     Test the input arguments
	lsvec = jobu == 'U' || jobu == 'F'
	jracc = jobv == 'J'
	rsvec = jobv == 'V' || jracc
	rowpiv = joba == 'F' || joba == 'G'
	l2rank = joba == 'R'
	l2aber = joba == 'A'
	errest = joba == 'E' || joba == 'G'
	l2tran = jobt == 'T' && ((*m) == (*n))
	l2kill = jobr == 'R'
	defr = jobr == 'N'
	l2pert = jobp == 'P'

	lquery = ((*lwork) == -1) || ((*lrwork) == -1)

	if !(rowpiv || l2rank || l2aber || errest || joba == 'C') {
		(*info) = -1
	} else if !(lsvec || jobu == 'N' || (jobu == 'W' && rsvec && l2tran)) {
		(*info) = -2
	} else if !(rsvec || jobv == 'N' || (jobv == 'W' && lsvec && l2tran)) {
		(*info) = -3
	} else if !(l2kill || defr) {
		(*info) = -4
	} else if !(jobt == 'T' || jobt == 'N') {
		(*info) = -5
	} else if !(l2pert || jobp == 'N') {
		(*info) = -6
	} else if (*m) < 0 {
		(*info) = -7
	} else if ((*n) < 0) || ((*n) > (*m)) {
		(*info) = -8
	} else if (*lda) < (*m) {
		(*info) = -10
	} else if lsvec && ((*ldu) < (*m)) {
		(*info) = -13
	} else if rsvec && ((*ldv) < (*n)) {
		(*info) = -15
	} else {

		(*info) = 0
	}

	if (*info) == 0 {
		//         .. compute the minimal and the optimal workspace lengths
		//         [[The expressions for computing the minimal and the optimal
		//         values of LCWORK, LRWORK are written with a lot of redundancy and
		//         can be simplified. However, this verbose form is useful for
		//         maintenance and modifications of the code.]]
		//
		//        .. minimal workspace length for ZGEQP3 of an M x N matrix,
		//         ZGEQRF of an N x N matrix, ZGELQF of an N x N matrix,
		//         ZUNMLQ for computing N x N matrix, ZUNMQR for computing N x N
		//         matrix, ZUNMQR for computing M x N matrix, respectively.
		lwqp3 = (*n) + 1
		lwqrf = maxint(1, *n)
		lwlqf = maxint(1, *n)
		lwunmlq = maxint(1, *n)
		lwunmqr = maxint(1, *n)
		lwunmqrm = maxint(1, *m)
		//        .. minimal workspace length for ZPOCON of an N x N matrix
		lwcon = 2 * (*n)
		//        .. minimal workspace length for ZGESVJ of an N x N matrix,
		//         without and with explicit accumulation of Jacobi rotations
		lwsvdj = maxint(2*(*n), 1)
		lwsvdjv = maxint(2*(*n), 1)
		//         .. minimal REAL workspace length for ZGEQP3, ZPOCON, ZGESVJ
		lrwqp3 = 2 * (*n)
		lrwcon = (*n)
		lrwsvdj = (*n)
		if lquery {
			Zgeqp3(m, n, a, lda, iwork, cdummy, cdummy, toPtr(-1), rdummy, &ierr)
			lwrkZgeqp3 = int(cdummy.GetRe(0))
			Zgeqrf(n, n, a, lda, cdummy, cdummy, toPtr(-1), &ierr)
			lwrkZgeqrf = int(cdummy.GetRe(0))
			Zgelqf(n, n, a, lda, cdummy, cdummy, toPtr(-1), &ierr)
			lwrkZgelqf = int(cdummy.GetRe(0))
		}
		minwrk = 2
		optwrk = 2
		miniwrk = (*n)
		if !(lsvec || rsvec) {
			//             .. minimal and optimal sizes of the complex workspace if
			//             only the singular values are requested
			if errest {
				minwrk = maxint((*n)+lwqp3, powint(*n, 2)+lwcon, (*n)+lwqrf, lwsvdj)
			} else {
				minwrk = maxint((*n)+lwqp3, (*n)+lwqrf, lwsvdj)
			}
			if lquery {
				Zgesvj('L', 'N', 'N', n, n, a, lda, sva, n, v, ldv, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
				lwrkZgesvj = int(cdummy.GetRe(0))
				if errest {
					optwrk = maxint((*n)+lwrkZgeqp3, powint(*n, 2)+lwcon, (*n)+lwrkZgeqrf, lwrkZgesvj)
				} else {
					optwrk = maxint((*n)+lwrkZgeqp3, (*n)+lwrkZgeqrf, lwrkZgesvj)
				}
			}
			if l2tran || rowpiv {
				if errest {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwcon, lrwsvdj)
				} else {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj)
				}
			} else {
				if errest {
					minrwrk = maxint(7, lrwqp3, lrwcon, lrwsvdj)
				} else {
					minrwrk = maxint(7, lrwqp3, lrwsvdj)
				}
			}
			if rowpiv || l2tran {
				miniwrk = miniwrk + (*m)
			}
		} else if rsvec && (!lsvec) {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            singular values and the right singular vectors are requested
			if errest {
				minwrk = maxint((*n)+lwqp3, lwcon, lwsvdj, (*n)+lwlqf, 2*(*n)+lwqrf, (*n)+lwsvdj, (*n)+lwunmlq)
			} else {
				minwrk = maxint((*n)+lwqp3, lwsvdj, (*n)+lwlqf, 2*(*n)+lwqrf, (*n)+lwsvdj, (*n)+lwunmlq)
			}
			if lquery {
				Zgesvj('L', 'U', 'N', n, n, u, ldu, sva, n, a, lda, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
				lwrkZgesvj = int(cdummy.GetRe(0))
				Zunmlq('L', 'C', n, n, n, a, lda, cdummy, v, ldv, cdummy, toPtr(-1), &ierr)
				lwrkZunmlq = int(cdummy.GetRe(0))
				if errest {
					optwrk = maxint((*n)+lwrkZgeqp3, lwcon, lwrkZgesvj, (*n)+lwrkZgelqf, 2*(*n)+lwrkZgeqrf, (*n)+lwrkZgesvj, (*n)+lwrkZunmlq)
				} else {
					optwrk = maxint((*n)+lwrkZgeqp3, lwrkZgesvj, (*n)+lwrkZgelqf, 2*(*n)+lwrkZgeqrf, (*n)+lwrkZgesvj, (*n)+lwrkZunmlq)
				}
			}
			if l2tran || rowpiv {
				if errest {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj, lrwcon)
				} else {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj)
				}
			} else {
				if errest {
					minrwrk = maxint(7, lrwqp3, lrwsvdj, lrwcon)
				} else {
					minrwrk = maxint(7, lrwqp3, lrwsvdj)
				}
			}
			if rowpiv || l2tran {
				miniwrk = miniwrk + (*m)
			}
		} else if lsvec && (!rsvec) {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            singular values and the left singular vectors are requested
			if errest {
				minwrk = (*n) + maxint(lwqp3, lwcon, (*n)+lwqrf, lwsvdj, lwunmqrm)
			} else {
				minwrk = (*n) + maxint(lwqp3, (*n)+lwqrf, lwsvdj, lwunmqrm)
			}
			if lquery {
				Zgesvj('L', 'U', 'N', n, n, u, ldu, sva, n, a, lda, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
				lwrkZgesvj = int(cdummy.GetRe(0))
				Zunmqr('L', 'N', m, n, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
				lwrkZunmqrm = int(cdummy.GetRe(0))
				if errest {
					optwrk = (*n) + maxint(lwrkZgeqp3, lwcon, (*n)+lwrkZgeqrf, lwrkZgesvj, lwrkZunmqrm)
				} else {
					optwrk = (*n) + maxint(lwrkZgeqp3, (*n)+lwrkZgeqrf, lwrkZgesvj, lwrkZunmqrm)
				}
			}
			if l2tran || rowpiv {
				if errest {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj, lrwcon)
				} else {
					minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj)
				}
			} else {
				if errest {
					minrwrk = maxint(7, lrwqp3, lrwsvdj, lrwcon)
				} else {
					minrwrk = maxint(7, lrwqp3, lrwsvdj)
				}
			}
			if rowpiv || l2tran {
				miniwrk = miniwrk + (*m)
			}
		} else {
			//            .. minimal and optimal sizes of the complex workspace if the
			//            full SVD is requested
			if !jracc {
				if errest {
					minwrk = maxint((*n)+lwqp3, (*n)+lwcon, 2*(*n)+powint(*n, 2)+lwcon, 2*(*n)+lwqrf, 2*(*n)+lwqp3, 2*(*n)+powint(*n, 2)+(*n)+lwlqf, 2*(*n)+powint(*n, 2)+(*n)+powint(*n, 2)+lwcon, 2*(*n)+powint(*n, 2)+(*n)+lwsvdj, 2*(*n)+powint(*n, 2)+(*n)+lwsvdjv, 2*(*n)+powint(*n, 2)+(*n)+lwunmqr, 2*(*n)+powint(*n, 2)+(*n)+lwunmlq, (*n)+powint(*n, 2)+lwsvdj, (*n)+lwunmqrm)
				} else {
					minwrk = maxint((*n)+lwqp3, 2*(*n)+powint(*n, 2)+lwcon, 2*(*n)+lwqrf, 2*(*n)+lwqp3, 2*(*n)+powint(*n, 2)+(*n)+lwlqf, 2*(*n)+powint(*n, 2)+(*n)+powint(*n, 2)+lwcon, 2*(*n)+powint(*n, 2)+(*n)+lwsvdj, 2*(*n)+powint(*n, 2)+(*n)+lwsvdjv, 2*(*n)+powint(*n, 2)+(*n)+lwunmqr, 2*(*n)+powint(*n, 2)+(*n)+lwunmlq, (*n)+powint(*n, 2)+lwsvdj, (*n)+lwunmqrm)
				}
				miniwrk = miniwrk + (*n)
				if rowpiv || l2tran {
					miniwrk = miniwrk + (*m)
				}
			} else {
				if errest {
					minwrk = maxint((*n)+lwqp3, (*n)+lwcon, 2*(*n)+lwqrf, 2*(*n)+powint(*n, 2)+lwsvdjv, 2*(*n)+powint(*n, 2)+(*n)+lwunmqr, (*n)+lwunmqrm)
				} else {
					minwrk = maxint((*n)+lwqp3, 2*(*n)+lwqrf, 2*(*n)+powint(*n, 2)+lwsvdjv, 2*(*n)+powint(*n, 2)+(*n)+lwunmqr, (*n)+lwunmqrm)
				}
				if rowpiv || l2tran {
					miniwrk = miniwrk + (*m)
				}
			}
			if lquery {
				Zunmqr('L', 'N', m, n, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
				lwrkZunmqrm = int(cdummy.GetRe(0))
				Zunmqr('L', 'N', n, n, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
				lwrkZunmqr = int(cdummy.GetRe(0))
				if !jracc {
					Zgeqp3(n, n, a, lda, iwork, cdummy, cdummy, toPtr(-1), rdummy, &ierr)
					lwrkZgeqp3n = int(cdummy.GetRe(0))
					Zgesvj('L', 'U', 'N', n, n, u, ldu, sva, n, v, ldv, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
					lwrkZgesvj = int(cdummy.GetRe(0))
					Zgesvj('U', 'U', 'N', n, n, u, ldu, sva, n, v, ldv, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
					lwrkZgesvju = int(cdummy.GetRe(0))
					Zgesvj('L', 'U', 'V', n, n, u, ldu, sva, n, v, ldv, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
					lwrkZgesvjv = int(cdummy.GetRe(0))
					Zunmlq('L', 'C', n, n, n, a, lda, cdummy, v, ldv, cdummy, toPtr(-1), &ierr)
					lwrkZunmlq = int(cdummy.GetRe(0))
					if errest {
						optwrk = maxint((*n)+lwrkZgeqp3, (*n)+lwcon, 2*(*n)+powint(*n, 2)+lwcon, 2*(*n)+lwrkZgeqrf, 2*(*n)+lwrkZgeqp3n, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgelqf, 2*(*n)+powint(*n, 2)+(*n)+powint(*n, 2)+lwcon, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgesvj, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgesvjv, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmqr, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmlq, (*n)+powint(*n, 2)+lwrkZgesvju, (*n)+lwrkZunmqrm)
					} else {
						optwrk = maxint((*n)+lwrkZgeqp3, 2*(*n)+powint(*n, 2)+lwcon, 2*(*n)+lwrkZgeqrf, 2*(*n)+lwrkZgeqp3n, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgelqf, 2*(*n)+powint(*n, 2)+(*n)+powint(*n, 2)+lwcon, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgesvj, 2*(*n)+powint(*n, 2)+(*n)+lwrkZgesvjv, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmqr, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmlq, (*n)+powint(*n, 2)+lwrkZgesvju, (*n)+lwrkZunmqrm)
					}
				} else {
					Zgesvj('L', 'U', 'V', n, n, u, ldu, sva, n, v, ldv, cdummy, toPtr(-1), rdummy, toPtr(-1), &ierr)
					lwrkZgesvjv = int(cdummy.GetRe(0))
					Zunmqr('L', 'N', n, n, n, cdummy.CMatrix(*n, opts), n, cdummy, v, ldv, cdummy, toPtr(-1), &ierr)
					lwrkZunmqr = int(cdummy.GetRe(0))
					Zunmqr('L', 'N', m, n, n, a, lda, cdummy, u, ldu, cdummy, toPtr(-1), &ierr)
					lwrkZunmqrm = int(cdummy.GetRe(0))
					if errest {
						optwrk = maxint((*n)+lwrkZgeqp3, (*n)+lwcon, 2*(*n)+lwrkZgeqrf, 2*(*n)+powint(*n, 2), 2*(*n)+powint(*n, 2)+lwrkZgesvjv, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmqr, (*n)+lwrkZunmqrm)
					} else {
						optwrk = maxint((*n)+lwrkZgeqp3, 2*(*n)+lwrkZgeqrf, 2*(*n)+powint(*n, 2), 2*(*n)+powint(*n, 2)+lwrkZgesvjv, 2*(*n)+powint(*n, 2)+(*n)+lwrkZunmqr, (*n)+lwrkZunmqrm)
					}
				}
			}
			if l2tran || rowpiv {
				minrwrk = maxint(7, 2*(*m), lrwqp3, lrwsvdj, lrwcon)
			} else {
				minrwrk = maxint(7, lrwqp3, lrwsvdj, lrwcon)
			}
		}
		minwrk = maxint(2, minwrk)
		optwrk = maxint(minwrk, optwrk)
		if (*lwork) < minwrk && (!lquery) {
			(*info) = -17
		}
		if (*lrwork) < minrwrk && (!lquery) {
			(*info) = -19
		}
	}

	if (*info) != 0 {

		gltest.Xerbla([]byte("ZGEJSV"), -(*info))
		return
	} else if lquery {
		cwork.SetRe(0, float64(optwrk))
		cwork.SetRe(1, float64(minwrk))
		rwork.Set(0, float64(minrwrk))
		(*iwork)[0] = maxint(4, miniwrk)
		return
	}

	//     Quick return for void matrix (Y3K safe)
	if ((*m) == 0) || ((*n) == 0) {
		(*iwork)[0] = 0
		rwork.Set(0, 0)
		return
	}

	//     Determine whether the matrix U should be M x N or M x M
	if lsvec {
		n1 = (*n)
		if jobu == 'F' {
			n1 = (*m)
		}
	}

	//     Set numerical parameters
	//
	//!    NOTE: Make sure DLAMCH() does not fail on the target architecture.
	epsln = Dlamch(Epsilon)
	sfmin = Dlamch(SafeMinimum)
	small = sfmin / epsln
	big = Dlamch(Overflow)
	//     BIG   = ONE / SFMIN
	//
	//     Initialize SVA(1:N) = diag( ||A e_i||_2 )_1^N
	//
	//(!)  If necessary, scale SVA() to protect the largest norm from
	//     overflow. It is possible that this scaling pushes the smallest
	//     column norm left from the underflow threshold (extreme case).
	scalem = one / math.Sqrt(float64(*m)*float64(*n))
	noscal = true
	goscal = true
	for p = 1; p <= (*n); p++ {
		aapp = zero
		aaqq = one
		Zlassq(m, a.CVector(0, p-1), func() *int { y := 1; return &y }(), &aapp, &aaqq)
		if aapp > big {
			(*info) = -9
			gltest.Xerbla([]byte("ZGEJSV"), -(*info))
			return
		}
		aaqq = math.Sqrt(aaqq)
		if (aapp < (big / aaqq)) && noscal {
			sva.Set(p-1, aapp*aaqq)
		} else {
			noscal = false
			sva.Set(p-1, aapp*(aaqq*scalem))
			if goscal {
				goscal = false
				goblas.Dscal(toPtr(p-1), &scalem, sva, func() *int { y := 1; return &y }())
			}
		}
	}

	if noscal {
		scalem = one
	}

	aapp = zero
	aaqq = big
	for p = 1; p <= (*n); p++ {
		aapp = maxf64(aapp, sva.Get(p-1))
		if sva.Get(p-1) != zero {
			aaqq = minf64(aaqq, sva.Get(p-1))
		}
	}

	//     Quick return for zero M x N matrix
	if aapp == zero {
		if lsvec {
			Zlaset('G', m, &n1, &czero, &cone, u, ldu)
		}
		if rsvec {
			Zlaset('G', n, n, &czero, &cone, v, ldv)
		}
		rwork.Set(0, one)
		rwork.Set(1, one)
		if errest {
			rwork.Set(2, one)
		}
		if lsvec && rsvec {
			rwork.Set(3, one)
			rwork.Set(4, one)
		}
		if l2tran {
			rwork.Set(5, zero)
			rwork.Set(6, zero)
		}
		(*iwork)[0] = 0
		(*iwork)[1] = 0
		(*iwork)[2] = 0
		(*iwork)[3] = -1
		return
	}

	//     Issue warning if denormalized column norms detected. Override the
	//     high relative accuracy request. Issue licence to kill nonzero columns
	//     (set them to zero) whose norm is less than sigma_max / BIG (roughly).
	warning = 0
	if aaqq <= sfmin {
		l2rank = true
		l2kill = true
		warning = 1
	}

	//     Quick return for one-column matrix
	if (*n) == 1 {

		if lsvec {
			Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), sva.GetPtr(0), &scalem, m, func() *int { y := 1; return &y }(), a, lda, &ierr)
			Zlacpy('A', m, func() *int { y := 1; return &y }(), a, lda, u, ldu)
			//           computing all M left singular vectors of the M x 1 matrix
			if n1 != (*n) {
				Zgeqrf(m, n, u, ldu, cwork, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
				Zungqr(m, &n1, func() *int { y := 1; return &y }(), u, ldu, cwork, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
				goblas.Zcopy(m, a.CVector(0, 0), func() *int { y := 1; return &y }(), u.CVector(0, 0), func() *int { y := 1; return &y }())
			}
		}
		if rsvec {
			v.Set(0, 0, cone)
		}
		if sva.Get(0) < (big * scalem) {
			sva.Set(0, sva.Get(0)/scalem)
			scalem = one
		}
		rwork.Set(0, one/scalem)
		rwork.Set(1, one)
		if sva.Get(0) != zero {
			(*iwork)[0] = 1
			if (sva.Get(0) / scalem) >= sfmin {
				(*iwork)[1] = 1
			} else {
				(*iwork)[1] = 0
			}
		} else {
			(*iwork)[0] = 0
			(*iwork)[1] = 0
		}
		(*iwork)[2] = 0
		(*iwork)[3] = -1
		if errest {
			rwork.Set(2, one)
		}
		if lsvec && rsvec {
			rwork.Set(3, one)
			rwork.Set(4, one)
		}
		if l2tran {
			rwork.Set(5, zero)
			rwork.Set(6, zero)
		}
		return

	}

	transp = false

	aatmax = -one
	aatmin = big
	if rowpiv || l2tran {
		//     Compute the row norms, needed to determine row pivoting sequence
		//     (in the case of heavily row weighted A, row pivoting is strongly
		//     advised) and to collect information needed to compare the
		//     structures of A * A^* and A^* * A (in the case L2TRAN.EQ..TRUE.).
		if l2tran {
			for p = 1; p <= (*m); p++ {
				xsc = zero
				temp1 = one
				Zlassq(n, a.CVector(p-1, 0), lda, &xsc, &temp1)
				//              ZLASSQ gets both the ell_2 and the ell_infinity norm
				//              in one pass through the vector
				rwork.Set((*m)+p-1, xsc*scalem)
				rwork.Set(p-1, xsc*(scalem*math.Sqrt(temp1)))
				aatmax = maxf64(aatmax, rwork.Get(p-1))
				if rwork.Get(p-1) != zero {
					aatmin = minf64(aatmin, rwork.Get(p-1))
				}
			}
		} else {
			for p = 1; p <= (*m); p++ {
				rwork.Set((*m)+p-1, scalem*a.GetMag(p-1, goblas.Izamax(n, a.CVector(p-1, 0), lda)-1))
				aatmax = maxf64(aatmax, rwork.Get((*m)+p-1))
				aatmin = minf64(aatmin, rwork.Get((*m)+p-1))
			}
		}

	}

	//     For square matrix A try to determine whether A^*  would be better
	//     input for the preconditioned Jacobi SVD, with faster convergence.
	//     The decision is based on an O(N) function of the vector of column
	//     and row norms of A, based on the Shannon entropy. This should give
	//     the right choice in most cases when the difference actually matters.
	//     It may fail and pick the slower converging side.
	entra = zero
	entrat = zero
	if l2tran {

		xsc = zero
		temp1 = one
		Dlassq(n, sva, func() *int { y := 1; return &y }(), &xsc, &temp1)
		temp1 = one / temp1

		entra = zero
		for p = 1; p <= (*n); p++ {
			big1 = math.Pow(sva.Get(p-1)/xsc, 2) * temp1
			if big1 != zero {
				entra = entra + big1*math.Log(big1)
			}
		}
		entra = -entra / math.Log(float64(*n))

		//        Now, SVA().^2/Trace(A^* * A) is a point in the probability simplex.
		//        It is derived from the diagonal of  A^* * A.  Do the same with the
		//        diagonal of A * A^*, compute the entropy of the corresponding
		//        probability distribution. Note that A * A^* and A^* * A have the
		//        same trace.
		entrat = zero
		for p = 1; p <= (*m); p++ {
			big1 = math.Pow(rwork.Get(p-1)/xsc, 2) * temp1
			if big1 != zero {
				entrat = entrat + big1*math.Log(big1)
			}
		}
		entrat = -entrat / math.Log(float64(*m))

		//        Analyze the entropies and decide A or A^*. Smaller entropy
		//        usually means better input for the algorithm.
		transp = (entrat < entra)

		//        If A^* is better than A, take the adjoint of A. This is allowed
		//        only for square matrices, M=N.
		if transp {
			//           In an optimal implementation, this trivial transpose
			//           should be replaced with faster transpose.
			for p = 1; p <= (*n)-1; p++ {
				a.Set(p-1, p-1, a.GetConj(p-1, p-1))
				for q = p + 1; q <= (*n); q++ {
					ctemp = a.GetConj(q-1, p-1)
					a.Set(q-1, p-1, a.GetConj(p-1, q-1))
					a.Set(p-1, q-1, ctemp)
				}
			}
			a.Set((*n)-1, (*n)-1, a.GetConj((*n)-1, (*n)-1))
			for p = 1; p <= (*n); p++ {
				rwork.Set((*m)+p-1, sva.Get(p-1))
				sva.Set(p-1, rwork.Get(p-1))
				//              previously computed row 2-norms are now column 2-norms
				//              of the transposed matrix
			}
			temp1 = aapp
			aapp = aatmax
			aatmax = temp1
			temp1 = aaqq
			aaqq = aatmin
			aatmin = temp1
			kill = lsvec
			lsvec = rsvec
			rsvec = kill
			if lsvec {
				n1 = (*n)
			}

			rowpiv = true
		}

	}
	//     END IF L2TRAN
	//
	//     Scale the matrix so that its maximal singular value remains less
	//     than SQRT(BIG) -- the matrix is scaled so that its maximal column
	//     has Euclidean norm equal to SQRT(BIG/N). The only reason to keep
	//     SQRT(BIG) instead of BIG is the fact that ZGEJSV uses LAPACK and
	//     BLAS routines that, in some implementations, are not capable of
	//     working in the full interval [SFMIN,BIG] and that they may provoke
	//     overflows in the intermediate results. If the singular values spread
	//     from SFMIN to BIG, then ZGESVJ will compute them. So, in that case,
	//     one should use ZGESVJ instead of ZGEJSV.
	//     >> change in the April 2016 update: allow bigger range, i.e. the
	//     largest column is allowed up to BIG/N and ZGESVJ will do the rest.
	big1 = math.Sqrt(big)
	temp1 = math.Sqrt(big / float64(*n))
	//      TEMP1  = BIG/DBLE(N)
	Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &temp1, n, func() *int { y := 1; return &y }(), sva.Matrix(*n, opts), n, &ierr)
	if aaqq > (aapp * sfmin) {
		aaqq = (aaqq / aapp) * temp1
	} else {
		aaqq = (aaqq * temp1) / aapp
	}
	temp1 = temp1 * scalem
	Zlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &aapp, &temp1, m, n, a, lda, &ierr)

	//     To undo scaling at the end of this procedure, multiply the
	//     computed singular values with USCAL2 / USCAL1.
	uscal1 = temp1
	uscal2 = aapp

	if l2kill {
		//        L2KILL enforces computation of nonzero singular values in
		//        the restricted range of condition number of the initial A,
		//        sigma_max(A) / sigma_min(A) approx. SQRT(BIG)/SQRT(SFMIN).
		xsc = math.Sqrt(sfmin)
	} else {
		xsc = small

		//        Now, if the condition number of A is too big,
		//        sigma_max(A) / sigma_min(A) .GT. SQRT(BIG/N) * EPSLN / SFMIN,
		//        as a precaution measure, the full SVD is computed using ZGESVJ
		//        with accumulated Jacobi rotations. This provides numerically
		//        more robust computation, at the cost of slightly increased run
		//        time. Depending on the concrete implementation of BLAS and LAPACK
		//        (i.e. how they behave in presence of extreme ill-conditioning) the
		//        implementor may decide to remove this switch.
		if (aaqq < math.Sqrt(sfmin)) && lsvec && rsvec {
			jracc = true
		}

	}
	if aaqq < xsc {
		for p = 1; p <= (*n); p++ {
			if sva.Get(p-1) < xsc {
				Zlaset('A', m, func() *int { y := 1; return &y }(), &czero, &czero, a.Off(0, p-1), lda)
				sva.Set(p-1, zero)
			}
		}
	}

	//     Preconditioning using QR factorization with pivoting
	if rowpiv {
		//        Optional row permutation (Bjoerck row pivoting):
		//        A result by Cox and Higham shows that the Bjoerck's
		//        row pivoting combined with standard column pivoting
		//        has similar effect as Powell-Reid complete pivoting.
		//        The ell-infinity norms of A are made nonincreasing.
		if (lsvec && rsvec) && !jracc {
			iwoff = 2 * (*n)
		} else {
			iwoff = (*n)
		}
		for p = 1; p <= (*m)-1; p++ {
			q = goblas.Idamax(toPtr((*m)-p+1), rwork.Off((*m)+p-1), func() *int { y := 1; return &y }()) + p - 1
			(*iwork)[iwoff+p-1] = q
			if p != q {
				temp1 = rwork.Get((*m) + p - 1)
				rwork.Set((*m)+p-1, rwork.Get((*m)+q-1))
				rwork.Set((*m)+q-1, temp1)
			}
		}
		Zlaswp(n, a, lda, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, iwoff+1-1), func() *int { y := 1; return &y }())
	}

	//     End of the preparation phase (scaling, optional sorting and
	//     transposing, optional flushing of small columns).
	//
	//     Preconditioning
	//
	//     If the full SVD is needed, the right singular vectors are computed
	//     from a matrix equation, and for that we need theoretical analysis
	//     of the Businger-Golub pivoting. So we use ZGEQP3 as the first RR QRF.
	//     In all other cases the first RR QRF can be chosen by other criteria
	//     (eg speed by replacing global with restricted window pivoting, such
	//     as in xGEQPX from TOMS # 782). Good results will be obtained using
	//     xGEQPX with properly (!) chosen numerical parameters.
	//     Any improvement of ZGEQP3 improves overal performance of ZGEJSV.
	//
	//     A * P1 = Q1 * [ R1^* 0]^*:
	for p = 1; p <= (*n); p++ {
		//        .. all columns are free columns
		(*iwork)[p-1] = 0
	}
	Zgeqp3(m, n, a, lda, iwork, cwork, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), rwork, &ierr)

	//     The upper triangular matrix R1 from the first QRF is inspected for
	//     rank deficiency and possibilities for deflation, or possible
	//     ill-conditioning. Depending on the user specified flag L2RANK,
	//     the procedure explores possibilities to reduce the numerical
	//     rank by inspecting the computed upper triangular factor. If
	//     L2RANK or L2ABER are up, then ZGEJSV will compute the SVD of
	//     A + dA, where ||dA|| <= f(M,N)*EPSLN.
	nr = 1
	if l2aber {
		//        Standard absolute error bound suffices. All sigma_i with
		//        sigma_i < N*EPSLN*||A|| are flushed to zero. This is an
		//        aggressive enforcement of lower numerical rank by introducing a
		//        backward error of the order of N*EPSLN*||A||.
		temp1 = math.Sqrt(float64(*n)) * epsln
		for p = 2; p <= (*n); p++ {
			if a.GetMag(p-1, p-1) >= (temp1 * a.GetMag(0, 0)) {
				nr = nr + 1
			} else {
				goto label3002
			}
		}
	label3002:
	} else if l2rank {
		//        .. similarly as above, only slightly more gentle (less aggressive).
		//        Sudden drop on the diagonal of R1 is used as the criterion for
		//        close-to-rank-deficient.
		temp1 = math.Sqrt(sfmin)
		for p = 2; p <= (*n); p++ {
			if (a.GetMag(p-1, p-1) < (epsln * a.GetMag(p-1-1, p-1-1))) || (a.GetMag(p-1, p-1) < small) || (l2kill && (a.GetMag(p-1, p-1) < temp1)) {
				goto label3402
			}
			nr = nr + 1
		}
	label3402:
	} else {
		//        The goal is high relative accuracy. However, if the matrix
		//        has high scaled condition number the relative accuracy is in
		//        general not feasible. Later on, a condition number estimator
		//        will be deployed to estimate the scaled condition number.
		//        Here we just remove the underflowed part of the triangular
		//        factor. This prevents the situation in which the code is
		//        working hard to get the accuracy not warranted by the data.
		temp1 = math.Sqrt(sfmin)
		for p = 2; p <= (*n); p++ {
			if (a.GetMag(p-1, p-1) < small) || (l2kill && (a.GetMag(p-1, p-1) < temp1)) {
				goto label3302
			}
			nr = nr + 1
		}
	label3302:
	}

	almort = false
	if nr == (*n) {
		maxprj = one
		for p = 2; p <= (*n); p++ {
			temp1 = a.GetMag(p-1, p-1) / sva.Get((*iwork)[p-1]-1)
			maxprj = minf64(maxprj, temp1)
		}
		if math.Pow(maxprj, 2) >= one-float64(*n)*epsln {
			almort = true
		}
	}

	sconda = -one
	condr1 = -one
	condr2 = -one

	if errest {
		if (*n) == nr {
			if rsvec {
				//              .. V is available as workspace
				Zlacpy('U', n, n, a, lda, v, ldv)
				for p = 1; p <= (*n); p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					goblas.Zdscal(&p, toPtrf64(one/temp1), v.CVector(0, p-1), func() *int { y := 1; return &y }())
				}
				if lsvec {
					Zpocon('U', n, v, ldv, &one, &temp1, cwork.Off((*n)+1-1), rwork, &ierr)
				} else {
					Zpocon('U', n, v, ldv, &one, &temp1, cwork, rwork, &ierr)
				}

			} else if lsvec {
				//              .. U is available as workspace
				Zlacpy('U', n, n, a, lda, u, ldu)
				for p = 1; p <= (*n); p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					goblas.Zdscal(&p, toPtrf64(one/temp1), u.CVector(0, p-1), func() *int { y := 1; return &y }())
				}
				Zpocon('U', n, u, ldu, &one, &temp1, cwork.Off((*n)+1-1), rwork, &ierr)
			} else {
				Zlacpy('U', n, n, a, lda, cwork.CMatrix(*n, opts), n)
				//[]            CALL ZLACPY( 'U', N, N, A, LDA, CWORK(N+1), N )
				//              Change: here index shifted by N to the left, CWORK(1:N)
				//              not needed for SIGMA only computation
				for p = 1; p <= (*n); p++ {
					temp1 = sva.Get((*iwork)[p-1] - 1)
					//[]               CALL ZDSCAL( p, ONE/TEMP1, CWORK(N+(p-1)*N+1), 1 )
					goblas.Zdscal(&p, toPtrf64(one/temp1), cwork.Off((p-1)*(*n)+1-1), func() *int { y := 1; return &y }())
				}
				//           .. the columns of R are scaled to have unit Euclidean lengths.
				//[]               CALL ZPOCON( 'U', N, CWORK(N+1), N, ONE, TEMP1,
				//[]     $              CWORK(N+N*N+1), RWORK, IERR )
				Zpocon('U', n, cwork.CMatrix(*n, opts), n, &one, &temp1, cwork.Off((*n)*(*n)+1-1), rwork, &ierr)

			}
			if temp1 != zero {
				sconda = one / math.Sqrt(temp1)
			} else {
				sconda = -one
			}
			//           SCONDA is an estimate of SQRT(||(R^* * R)^(-1)||_1).
			//           N^(-1/4) * SCONDA <= ||R^(-1)||_2 <= N^(1/4) * SCONDA
		} else {
			sconda = -one
		}
	}

	l2pert = l2pert && (cmplx.Abs(a.Get(0, 0)/a.Get(nr-1, nr-1)) > math.Sqrt(big1))
	//     If there is no violent scaling, artificial perturbation is not needed.
	//
	//     Phase 3:
	if !(rsvec || lsvec) {
		//         Singular Values only
		//
		//         .. transpose A(1:NR,1:N)
		for p = 1; p <= minint((*n)-1, nr); p++ {
			goblas.Zcopy(toPtr((*n)-p), a.CVector(p-1, p+1-1), lda, a.CVector(p+1-1, p-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr((*n)-p+1), a.CVector(p-1, p-1), func() *int { y := 1; return &y }())
		}
		if nr == (*n) {
			a.Set((*n)-1, (*n)-1, a.GetConj((*n)-1, (*n)-1))
		}

		//        The following two DO-loops introduce small relative perturbation
		//        into the strict upper triangle of the lower triangular matrix.
		//        Small entries below the main diagonal are also changed.
		//        This modification is useful if the computing environment does not
		//        provide/allow FLUSH TO ZERO underflow, for it prevents many
		//        annoying denormalized numbers in case of strongly scaled matrices.
		//        The perturbation is structured so that it does not introduce any
		//        new perturbation of the singular values, and it does not destroy
		//        the job done by the preconditioner.
		//        The licence for this perturbation is in the variable L2PERT, which
		//        should be .FALSE. if FLUSH TO ZERO underflow is active.
		if !almort {

			if l2pert {
				//              XSC = SQRT(SMALL)
				xsc = epsln / float64(*n)
				for q = 1; q <= nr; q++ {
					ctemp = complex(xsc*a.GetMag(q-1, q-1), zero)
					for p = 1; p <= (*n); p++ {
						if ((p > q) && (a.GetMag(p-1, q-1) <= temp1)) || (p < q) {
						}
						//     $                     A(p,q) = TEMP1 * ( A(p,q) / ABS(A(p,q)) )
						a.Set(p-1, q-1, ctemp)
					}
				}
			} else {
				Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, a.Off(0, 1), lda)
			}

			//            .. second preconditioning using the QR factorization
			Zgeqrf(n, &nr, a, lda, cwork, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			//           .. and transpose upper to lower triangular
			for p = 1; p <= nr-1; p++ {
				goblas.Zcopy(toPtr(nr-p), a.CVector(p-1, p+1-1), lda, a.CVector(p+1-1, p-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(nr-p+1), a.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			}

		}

		//           Row-cyclic Jacobi SVD algorithm with column pivoting
		//
		//           .. again some perturbation (a "background noise") is added
		//           to drown denormals
		if l2pert {
			//              XSC = SQRT(SMALL)
			xsc = epsln / float64(*n)
			for q = 1; q <= nr; q++ {
				ctemp = complex(xsc*a.GetMag(q-1, q-1), zero)
				for p = 1; p <= nr; p++ {
					if ((p > q) && (a.GetMag(p-1, q-1) <= temp1)) || (p < q) {
					}
					//     $                   A(p,q) = TEMP1 * ( A(p,q) / ABS(A(p,q)) )
					a.Set(p-1, q-1, ctemp)
				}
			}
		} else {
			Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, a.Off(0, 1), lda)
		}

		//           .. and one-sided Jacobi rotations are started on a lower
		//           triangular matrix (plus perturbation which is ignored in
		//           the part which destroys triangular form (confusing?!))
		Zgesvj('L', 'N', 'N', &nr, &nr, a, lda, sva, n, v, ldv, cwork, lwork, rwork, lrwork, info)

		scalem = rwork.Get(0)
		numrank = int(math.Round(rwork.Get(1)))

	} else if (rsvec && (!lsvec) && (!jracc)) || (jracc && (!lsvec) && (nr != (*n))) {
		//        -> Singular Values and Right Singular Vectors <-
		if almort {
			//           .. in this case NR equals N
			for p = 1; p <= nr; p++ {
				goblas.Zcopy(toPtr((*n)-p+1), a.CVector(p-1, p-1), lda, v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-p+1), v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			}
			Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)

			Zgesvj('L', 'U', 'N', n, &nr, v, ldv, sva, &nr, a, lda, cwork, lwork, rwork, lrwork, info)
			scalem = rwork.Get(0)
			numrank = int(math.Round(rwork.Get(1)))
		} else {
			//        .. two more QR factorizations ( one QRF is not enough, two require
			//        accumulated product of Jacobi rotations, three are perfect )
			Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, a.Off(1, 0), lda)
			Zgelqf(&nr, n, a, lda, cwork, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
			Zlacpy('L', &nr, &nr, a, lda, v, ldv)
			Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
			Zgeqrf(&nr, &nr, v, ldv, cwork.Off((*n)+1-1), cwork.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), &ierr)
			for p = 1; p <= nr; p++ {
				goblas.Zcopy(toPtr(nr-p+1), v.CVector(p-1, p-1), ldv, v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(nr-p+1), v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			}
			Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)

			Zgesvj('L', 'U', 'N', &nr, &nr, v, ldv, sva, &nr, u, ldu, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), rwork, lrwork, info)
			scalem = rwork.Get(0)
			numrank = int(math.Round(rwork.Get(1)))
			if nr < (*n) {
				Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
				Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
				Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
			}

			Zunmlq('L', 'C', n, n, &nr, a, lda, cwork, v, ldv, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

		}
		//         .. permute the rows of V
		//         DO 8991 p = 1, N
		//            CALL ZCOPY( N, V(p,1), LDV, A(IWORK(p),1), LDA )
		// 8991    CONTINUE
		//         CALL ZLACPY( 'All', N, N, A, LDA, V, LDV )
		Zlapmr(false, n, n, v, ldv, iwork)

		if transp {
			Zlacpy('A', n, n, v, ldv, u, ldu)
		}

	} else if jracc && (!lsvec) && (nr == (*n)) {

		Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, a.Off(1, 0), lda)

		Zgesvj('U', 'N', 'V', n, n, a, lda, sva, n, v, ldv, cwork, lwork, rwork, lrwork, info)
		scalem = rwork.Get(0)
		numrank = int(math.Round(rwork.Get(1)))
		Zlapmr(false, n, n, v, ldv, iwork)

	} else if lsvec && (!rsvec) {
		//        .. Singular Values and Left Singular Vectors                 ..
		//
		//        .. second preconditioning step to avoid need to accumulate
		//        Jacobi rotations in the Jacobi iterations.
		for p = 1; p <= nr; p++ {
			goblas.Zcopy(toPtr((*n)-p+1), a.CVector(p-1, p-1), lda, u.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr((*n)-p+1), u.CVector(p-1, p-1), func() *int { y := 1; return &y }())
		}
		Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(0, 1), ldu)

		Zgeqrf(n, &nr, u, ldu, cwork.Off((*n)+1-1), cwork.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), &ierr)

		for p = 1; p <= nr-1; p++ {
			goblas.Zcopy(toPtr(nr-p), u.CVector(p-1, p+1-1), ldu, u.CVector(p+1-1, p-1), func() *int { y := 1; return &y }())
			Zlacgv(toPtr((*n)-p+1), u.CVector(p-1, p-1), func() *int { y := 1; return &y }())
		}
		Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(0, 1), ldu)

		Zgesvj('L', 'U', 'N', &nr, &nr, u, ldu, sva, &nr, a, lda, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), rwork, lrwork, info)
		scalem = rwork.Get(0)
		numrank = int(math.Round(rwork.Get(1)))

		if nr < (*m) {
			Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
			if nr < n1 {
				Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
				Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
			}
		}

		Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

		if rowpiv {
			Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, iwoff+1-1), toPtr(-1))
		}

		for p = 1; p <= n1; p++ {
			xsc = one / goblas.Dznrm2(m, u.CVector(0, p-1), func() *int { y := 1; return &y }())
			goblas.Zdscal(m, &xsc, u.CVector(0, p-1), func() *int { y := 1; return &y }())
		}

		if transp {
			Zlacpy('A', n, n, u, ldu, v, ldv)
		}

	} else {
		//        .. Full SVD ..
		if !jracc {
			//
			if !almort {
				//           Second Preconditioning Step (QRF [with pivoting])
				//           Note that the composition of TRANSPOSE, QRF and TRANSPOSE is
				//           equivalent to an LQF CALL. Since in many libraries the QRF
				//           seems to be better optimized than the LQF, we do explicit
				//           transpose and use the QRF. This is subject to changes in an
				//           optimized implementation of ZGEJSV.
				for p = 1; p <= nr; p++ {
					goblas.Zcopy(toPtr((*n)-p+1), a.CVector(p-1, p-1), lda, v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
					Zlacgv(toPtr((*n)-p+1), v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
				}

				//           .. the following two loops perturb small entries to avoid
				//           denormals in the second QR factorization, where they are
				//           as good as zeros. This is done to avoid painfully slow
				//           computation with denormals. The relative size of the perturbation
				//           is a parameter that can be changed by the implementer.
				//           This perturbation device will be obsolete on machines with
				//           properly implemented arithmetic.
				//           To switch it off, set L2PERT=.FALSE. To remove it from  the
				//           code, remove the action under L2PERT=.TRUE., leave the ELSE part.
				//           The following two loops should be blocked and fused with the
				//           transposed copy above.
				if l2pert {
					xsc = math.Sqrt(small)
					for q = 1; q <= nr; q++ {
						ctemp = complex(xsc*v.GetMag(q-1, q-1), zero)
						for p = 1; p <= (*n); p++ {
							if (p > q) && (v.GetMag(p-1, q-1) <= temp1) || (p < q) {
							}
							//     $                   V(p,q) = TEMP1 * ( V(p,q) / ABS(V(p,q)) )
							v.Set(p-1, q-1, ctemp)
							if p < q {
								v.Set(p-1, q-1, -v.Get(p-1, q-1))
							}
						}
					}
				} else {
					Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
				}

				//           Estimate the row scaled condition number of R1
				//           (If R1 is rectangular, N > NR, then the condition number
				//           of the leading NR x NR submatrix is estimated.)
				Zlacpy('L', &nr, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+1-1, nr, opts), &nr)
				for p = 1; p <= nr; p++ {
					temp1 = goblas.Dznrm2(toPtr(nr-p+1), cwork.Off(2*(*n)+(p-1)*nr+p-1), func() *int { y := 1; return &y }())
					goblas.Zdscal(toPtr(nr-p+1), toPtrf64(one/temp1), cwork.Off(2*(*n)+(p-1)*nr+p-1), func() *int { y := 1; return &y }())
				}
				Zpocon('L', &nr, cwork.CMatrixOff(2*(*n)+1-1, nr, opts), &nr, &one, &temp1, cwork.Off(2*(*n)+nr*nr+1-1), rwork, &ierr)
				condr1 = one / math.Sqrt(temp1)
				//           .. here need a second opinion on the condition number
				//           .. then assume worst case scenario
				//           R1 is OK for inverse <=> CONDR1 .LT. DBLE(N)
				//           more conservative    <=> CONDR1 .LT. SQRT(DBLE(N))
				condOk = math.Sqrt(math.Sqrt(float64(nr)))
				//[TP]       COND_OK is a tuning parameter.

				if condr1 < condOk {
					//              .. the second QRF without pivoting. Note: in an optimized
					//              implementation, this QRF should be implemented as the QRF
					//              of a lower triangular matrix.
					//              R1^* = Q2 * R2
					Zgeqrf(n, &nr, v, ldv, cwork.Off((*n)+1-1), cwork.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), &ierr)

					if l2pert {
						xsc = math.Sqrt(small) / epsln
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								ctemp = complex(xsc*minf64(v.GetMag(p-1, p-1), v.GetMag(q-1, q-1)), zero)
								if v.GetMag(q-1, p-1) <= temp1 {
								}
								//     $                     V(q,p) = TEMP1 * ( V(q,p) / ABS(V(q,p)) )
								v.Set(q-1, p-1, ctemp)
							}
						}
					}

					if nr != (*n) {
						Zlacpy('A', n, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n)
					}
					//              .. save ...
					//
					//           .. this transposed copy should be better than naive
					for p = 1; p <= nr-1; p++ {
						goblas.Zcopy(toPtr(nr-p), v.CVector(p-1, p+1-1), ldv, v.CVector(p+1-1, p-1), func() *int { y := 1; return &y }())
						Zlacgv(toPtr(nr-p+1), v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
					}
					v.Set(nr-1, nr-1, v.GetConj(nr-1, nr-1))

					condr2 = condr1

				} else {
					//              .. ill-conditioned case: second QRF with pivoting
					//              Note that windowed pivoting would be equally good
					//              numerically, and more run-time efficient. So, in
					//              an optimal implementation, the next call to ZGEQP3
					//              should be replaced with eg. CALL ZGEQPX (ACM TOMS #782)
					//              with properly (carefully) chosen parameters.
					//
					//              R1^* * P2 = Q2 * R2
					for p = 1; p <= nr; p++ {
						(*iwork)[(*n)+p-1] = 0
					}
					Zgeqp3(n, &nr, v, ldv, toSlice(iwork, (*n)+1-1), cwork.Off((*n)+1-1), cwork.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), rwork, &ierr)
					//*               CALL ZGEQRF( N, NR, V, LDV, CWORK(N+1), CWORK(2*N+1),
					//*     $              LWORK-2*N, IERR )
					if l2pert {
						xsc = math.Sqrt(small)
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								ctemp = complex(xsc*minf64(v.GetMag(p-1, p-1), v.GetMag(q-1, q-1)), zero)
								if v.GetMag(q-1, p-1) <= temp1 {
								}
								//     $                     V(q,p) = TEMP1 * ( V(q,p) / ABS(V(q,p)) )
								v.Set(q-1, p-1, ctemp)
							}
						}
					}

					Zlacpy('A', n, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n)

					if l2pert {
						xsc = math.Sqrt(small)
						for p = 2; p <= nr; p++ {
							for q = 1; q <= p-1; q++ {
								ctemp = complex(xsc*minf64(v.GetMag(p-1, p-1), v.GetMag(q-1, q-1)), zero)
								//                        V(p,q) = - TEMP1*( V(q,p) / ABS(V(q,p)) )
								v.Set(p-1, q-1, -ctemp)
							}
						}
					} else {
						Zlaset('L', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(1, 0), ldv)
					}
					//              Now, compute R2 = L3 * Q3, the LQ factorization.
					Zgelqf(&nr, &nr, v, ldv, cwork.Off(2*(*n)+(*n)*nr+1-1), cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)
					//              .. and estimate the condition number
					Zlacpy('L', &nr, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+(*n)*nr+nr+1-1, nr, opts), &nr)
					for p = 1; p <= nr; p++ {
						temp1 = goblas.Dznrm2(&p, cwork.Off(2*(*n)+(*n)*nr+nr+p-1), &nr)
						goblas.Zdscal(&p, toPtrf64(one/temp1), cwork.Off(2*(*n)+(*n)*nr+nr+p-1), &nr)
					}
					Zpocon('L', &nr, cwork.CMatrixOff(2*(*n)+(*n)*nr+nr+1-1, nr, opts), &nr, &one, &temp1, cwork.Off(2*(*n)+(*n)*nr+nr+nr*nr+1-1), rwork, &ierr)
					condr2 = one / math.Sqrt(temp1)

					if condr2 >= condOk {
						//                 .. save the Householder vectors used for Q3
						//                 (this overwrites the copy of R2, as it will not be
						//                 needed in this branch, but it does not overwritte the
						//                 Huseholder vectors of Q2.).
						Zlacpy('U', &nr, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n)
						//                 .. and the rest of the information on Q3 is in
						//                 WORK(2*N+N*NR+1:2*N+N*NR+N)
					}

				}

				if l2pert {
					xsc = math.Sqrt(small)
					for q = 2; q <= nr; q++ {
						ctemp = complex(xsc, 0) * v.Get(q-1, q-1)
						for p = 1; p <= q-1; p++ {
							//                     V(p,q) = - TEMP1*( V(p,q) / ABS(V(p,q)) )
							v.Set(p-1, q-1, -ctemp)
						}
					}
				} else {
					Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
				}

				//        Second preconditioning finished; continue with Jacobi SVD
				//        The input matrix is lower trinagular.
				//
				//        Recover the right singular vectors as solution of a well
				//        conditioned triangular matrix equation.
				if condr1 < condOk {

					Zgesvj('L', 'U', 'N', &nr, &nr, v, ldv, sva, &nr, u, ldu, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), rwork, lrwork, info)
					scalem = rwork.Get(0)
					numrank = int(math.Round(rwork.Get(1)))
					for p = 1; p <= nr; p++ {
						goblas.Zcopy(&nr, v.CVector(0, p-1), func() *int { y := 1; return &y }(), u.CVector(0, p-1), func() *int { y := 1; return &y }())
						goblas.Zdscal(&nr, sva.GetPtr(p-1), v.CVector(0, p-1), func() *int { y := 1; return &y }())
					}
					//        .. pick the right matrix equation and solve it

					if nr == (*n) {
						// :))             .. best case, R1 is inverted. The solution of this matrix
						//                 equation is Q2*V2 = the product of the Jacobi rotations
						//                 used in ZGESVJ, premultiplied with the orthogonal matrix
						//                 from the second QR factorization.
						goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, &nr, &nr, &cone, a, lda, v, ldv)
					} else {
						//                 .. R1 is well conditioned, but non-square. Adjoint of R2
						//                 is inverted to get the product of the Jacobi rotations
						//                 used in ZGESVJ. The Q-factor from the second QR
						//                 factorization is then built in explicitly.
						goblas.Ztrsm(Left, Upper, ConjTrans, NonUnit, &nr, &nr, &cone, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, v, ldv)
						if nr < (*n) {
							Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
							Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
							Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
						}
						Zunmqr('L', 'N', n, n, &nr, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, cwork.Off((*n)+1-1), v, ldv, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)
					}

				} else if condr2 < condOk {
					//              The matrix R2 is inverted. The solution of the matrix equation
					//              is Q3^* * V3 = the product of the Jacobi rotations (appplied to
					//              the lower triangular L3 from the LQ factorization of
					//              R2=L3*Q3), pre-multiplied with the transposed Q3.
					Zgesvj('L', 'U', 'N', &nr, &nr, v, ldv, sva, &nr, u, ldu, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), rwork, lrwork, info)
					scalem = rwork.Get(0)
					numrank = int(math.Round(rwork.Get(1)))
					for p = 1; p <= nr; p++ {
						goblas.Zcopy(&nr, v.CVector(0, p-1), func() *int { y := 1; return &y }(), u.CVector(0, p-1), func() *int { y := 1; return &y }())
						goblas.Zdscal(&nr, sva.GetPtr(p-1), u.CVector(0, p-1), func() *int { y := 1; return &y }())
					}
					goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, &nr, &nr, &cone, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, u, ldu)
					//              .. apply the permutation from the second QR factorization
					for q = 1; q <= nr; q++ {
						for p = 1; p <= nr; p++ {
							cwork.Set(2*(*n)+(*n)*nr+nr+(*iwork)[(*n)+p-1]-1, u.Get(p-1, q-1))
						}
						for p = 1; p <= nr; p++ {
							u.Set(p-1, q-1, cwork.Get(2*(*n)+(*n)*nr+nr+p-1))
						}
					}
					if nr < (*n) {
						Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
						Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
						Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
					}
					Zunmqr('L', 'N', n, n, &nr, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, cwork.Off((*n)+1-1), v, ldv, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)
				} else {
					//              Last line of defense.
					// #:(          This is a rather pathological case: no scaled condition
					//              improvement after two pivoted QR factorizations. Other
					//              possibility is that the rank revealing QR factorization
					//              or the condition estimator has failed, or the COND_OK
					//              is set very close to ONE (which is unnecessary). Normally,
					//              this branch should never be executed, but in rare cases of
					//              failure of the RRQR or condition estimator, the last line of
					//              defense ensures that ZGEJSV completes the task.
					//              Compute the full SVD of L3 using ZGESVJ with explicit
					//              accumulation of Jacobi rotations.
					Zgesvj('L', 'U', 'V', &nr, &nr, v, ldv, sva, &nr, u, ldu, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), rwork, lrwork, info)
					scalem = rwork.Get(0)
					numrank = int(math.Round(rwork.Get(1)))
					if nr < (*n) {
						Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
						Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
						Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
					}
					Zunmqr('L', 'N', n, n, &nr, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, cwork.Off((*n)+1-1), v, ldv, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)

					Zunmlq('L', 'C', &nr, &nr, &nr, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, cwork.Off(2*(*n)+(*n)*nr+1-1), u, ldu, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)
					for q = 1; q <= nr; q++ {
						for p = 1; p <= nr; p++ {
							cwork.Set(2*(*n)+(*n)*nr+nr+(*iwork)[(*n)+p-1]-1, u.Get(p-1, q-1))
						}
						for p = 1; p <= nr; p++ {
							u.Set(p-1, q-1, cwork.Get(2*(*n)+(*n)*nr+nr+p-1))
						}
					}

				}

				//           Permute the rows of V using the (column) permutation from the
				//           first QRF. Also, scale the columns to make them unit in
				//           Euclidean norm. This applies to all cases.
				temp1 = math.Sqrt(float64(*n)) * epsln
				for q = 1; q <= (*n); q++ {
					for p = 1; p <= (*n); p++ {
						cwork.Set(2*(*n)+(*n)*nr+nr+(*iwork)[p-1]-1, v.Get(p-1, q-1))
					}
					for p = 1; p <= (*n); p++ {
						v.Set(p-1, q-1, cwork.Get(2*(*n)+(*n)*nr+nr+p-1))
					}
					xsc = one / goblas.Dznrm2(n, v.CVector(0, q-1), func() *int { y := 1; return &y }())
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						goblas.Zdscal(n, &xsc, v.CVector(0, q-1), func() *int { y := 1; return &y }())
					}
				}
				//           At this moment, V contains the right singular vectors of A.
				//           Next, assemble the left singular vector matrix U (M x N).
				if nr < (*m) {
					Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
					if nr < n1 {
						Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
						Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
					}
				}

				//           The Q matrix from the first QRF is built into the left singular
				//           matrix U. This applies to all cases.
				//
				Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
				//           The columns of U are normalized. The cost is O(M*N) flops.
				temp1 = math.Sqrt(float64(*m)) * epsln
				for p = 1; p <= nr; p++ {
					xsc = one / goblas.Dznrm2(m, u.CVector(0, p-1), func() *int { y := 1; return &y }())
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						goblas.Zdscal(m, &xsc, u.CVector(0, p-1), func() *int { y := 1; return &y }())
					}
				}

				//           If the initial QRF is computed with row pivoting, the left
				//           singular vectors must be adjusted.
				if rowpiv {
					Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, iwoff+1-1), toPtr(-1))
				}

			} else {
				//        .. the initial matrix A has almost orthogonal columns and
				//        the second QRF is not needed
				Zlacpy('U', n, n, a, lda, cwork.CMatrixOff((*n)+1-1, *n, opts), n)
				if l2pert {
					xsc = math.Sqrt(small)
					for p = 2; p <= (*n); p++ {
						ctemp = complex(xsc, 0) * cwork.Get((*n)+(p-1)*(*n)+p-1)
						for q = 1; q <= p-1; q++ {
							//                     CWORK(N+(q-1)*N+p)=-TEMP1 * ( CWORK(N+(p-1)*N+q) /
							//     $                                        ABS(CWORK(N+(p-1)*N+q)) )
							cwork.Set((*n)+(q-1)*(*n)+p-1, -ctemp)
						}
					}
				} else {
					Zlaset('L', toPtr((*n)-1), toPtr((*n)-1), &czero, &czero, cwork.CMatrixOff((*n)+2-1, *n, opts), n)
				}

				Zgesvj('U', 'U', 'N', n, n, cwork.CMatrixOff((*n)+1-1, *n, opts), n, sva, n, u, ldu, cwork.Off((*n)+(*n)*(*n)+1-1), toPtr((*lwork)-(*n)-(*n)*(*n)), rwork, lrwork, info)

				scalem = rwork.Get(0)
				numrank = int(math.Round(rwork.Get(1)))
				for p = 1; p <= (*n); p++ {
					goblas.Zcopy(n, cwork.Off((*n)+(p-1)*(*n)+1-1), func() *int { y := 1; return &y }(), u.CVector(0, p-1), func() *int { y := 1; return &y }())
					goblas.Zdscal(n, sva.GetPtr(p-1), cwork.Off((*n)+(p-1)*(*n)+1-1), func() *int { y := 1; return &y }())
				}

				goblas.Ztrsm(Left, Upper, NoTrans, NonUnit, n, n, &cone, a, lda, cwork.CMatrixOff((*n)+1-1, *n, opts), n)
				for p = 1; p <= (*n); p++ {
					goblas.Zcopy(n, cwork.Off((*n)+p-1), n, v.CVector((*iwork)[p-1]-1, 0), ldv)
				}
				temp1 = math.Sqrt(float64(*n)) * epsln
				for p = 1; p <= (*n); p++ {
					xsc = one / goblas.Dznrm2(n, v.CVector(0, p-1), func() *int { y := 1; return &y }())
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						goblas.Zdscal(n, &xsc, v.CVector(0, p-1), func() *int { y := 1; return &y }())
					}
				}

				//           Assemble the left singular vector matrix U (M x N).
				if (*n) < (*m) {
					Zlaset('A', toPtr((*m)-(*n)), n, &czero, &czero, u.Off((*n)+1-1, 0), ldu)
					if (*n) < n1 {
						Zlaset('A', n, toPtr(n1-(*n)), &czero, &czero, u.Off(0, (*n)+1-1), ldu)
						Zlaset('A', toPtr((*m)-(*n)), toPtr(n1-(*n)), &czero, &cone, u.Off((*n)+1-1, (*n)+1-1), ldu)
					}
				}
				Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)
				temp1 = math.Sqrt(float64(*m)) * epsln
				for p = 1; p <= n1; p++ {
					xsc = one / goblas.Dznrm2(m, u.CVector(0, p-1), func() *int { y := 1; return &y }())
					if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
						goblas.Zdscal(m, &xsc, u.CVector(0, p-1), func() *int { y := 1; return &y }())
					}
				}

				if rowpiv {
					Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, iwoff+1-1), toPtr(-1))
				}

			}

			//        end of the  >> almost orthogonal case <<  in the full SVD
		} else {
			//        This branch deploys a preconditioned Jacobi SVD with explicitly
			//        accumulated rotations. It is included as optional, mainly for
			//        experimental purposes. It does perform well, and can also be used.
			//        In this implementation, this branch will be automatically activated
			//        if the  condition number sigma_max(A) / sigma_min(A) is predicted
			//        to be greater than the overflow threshold. This is because the
			//        a posteriori computation of the singular vectors assumes robust
			//        implementation of BLAS and some LAPACK procedures, capable of working
			//        in presence of extreme values, e.g. when the singular values spread from
			//        the underflow to the overflow threshold.
			for p = 1; p <= nr; p++ {
				goblas.Zcopy(toPtr((*n)-p+1), a.CVector(p-1, p-1), lda, v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr((*n)-p+1), v.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			}

			if l2pert {
				xsc = math.Sqrt(small / epsln)
				for q = 1; q <= nr; q++ {
					ctemp = complex(xsc*v.GetMag(q-1, q-1), zero)
					for p = 1; p <= (*n); p++ {
						if (p > q) && (v.GetMag(p-1, q-1) <= temp1) || (p < q) {
						}
						//     $                V(p,q) = TEMP1 * ( V(p,q) / ABS(V(p,q)) )
						v.Set(p-1, q-1, ctemp)
						if p < q {
							v.Set(p-1, q-1, -v.Get(p-1, q-1))
						}
					}
				}
			} else {
				Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, v.Off(0, 1), ldv)
			}
			Zgeqrf(n, &nr, v, ldv, cwork.Off((*n)+1-1), cwork.Off(2*(*n)+1-1), toPtr((*lwork)-2*(*n)), &ierr)
			Zlacpy('L', n, &nr, v, ldv, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n)

			for p = 1; p <= nr; p++ {
				goblas.Zcopy(toPtr(nr-p+1), v.CVector(p-1, p-1), ldv, u.CVector(p-1, p-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(nr-p+1), u.CVector(p-1, p-1), func() *int { y := 1; return &y }())
			}
			if l2pert {
				xsc = math.Sqrt(small / epsln)
				for q = 2; q <= nr; q++ {
					for p = 1; p <= q-1; p++ {
						ctemp = complex(xsc*minf64(u.GetMag(p-1, p-1), u.GetMag(q-1, q-1)), zero)
						//                  U(p,q) = - TEMP1 * ( U(q,p) / ABS(U(q,p)) )
						u.Set(p-1, q-1, -ctemp)
					}
				}
			} else {
				Zlaset('U', toPtr(nr-1), toPtr(nr-1), &czero, &czero, u.Off(0, 1), ldu)
			}
			Zgesvj('L', 'U', 'V', &nr, &nr, u, ldu, sva, n, v, ldv, cwork.Off(2*(*n)+(*n)*nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr), rwork, lrwork, info)
			scalem = rwork.Get(0)
			numrank = int(math.Round(rwork.Get(1)))
			if nr < (*n) {
				Zlaset('A', toPtr((*n)-nr), &nr, &czero, &czero, v.Off(nr+1-1, 0), ldv)
				Zlaset('A', &nr, toPtr((*n)-nr), &czero, &czero, v.Off(0, nr+1-1), ldv)
				Zlaset('A', toPtr((*n)-nr), toPtr((*n)-nr), &czero, &cone, v.Off(nr+1-1, nr+1-1), ldv)
			}
			Zunmqr('L', 'N', n, n, &nr, cwork.CMatrixOff(2*(*n)+1-1, *n, opts), n, cwork.Off((*n)+1-1), v, ldv, cwork.Off(2*(*n)+(*n)*nr+nr+1-1), toPtr((*lwork)-2*(*n)-(*n)*nr-nr), &ierr)

			//           Permute the rows of V using the (column) permutation from the
			//           first QRF. Also, scale the columns to make them unit in
			//           Euclidean norm. This applies to all cases.
			temp1 = math.Sqrt(float64(*n)) * epsln
			for q = 1; q <= (*n); q++ {
				for p = 1; p <= (*n); p++ {
					cwork.Set(2*(*n)+(*n)*nr+nr+(*iwork)[p-1]-1, v.Get(p-1, q-1))
				}
				for p = 1; p <= (*n); p++ {
					v.Set(p-1, q-1, cwork.Get(2*(*n)+(*n)*nr+nr+p-1))
				}
				xsc = one / goblas.Dznrm2(n, v.CVector(0, q-1), func() *int { y := 1; return &y }())
				if (xsc < (one - temp1)) || (xsc > (one + temp1)) {
					goblas.Zdscal(n, &xsc, v.CVector(0, q-1), func() *int { y := 1; return &y }())
				}
			}

			//           At this moment, V contains the right singular vectors of A.
			//           Next, assemble the left singular vector matrix U (M x N).
			if nr < (*m) {
				Zlaset('A', toPtr((*m)-nr), &nr, &czero, &czero, u.Off(nr+1-1, 0), ldu)
				if nr < n1 {
					Zlaset('A', &nr, toPtr(n1-nr), &czero, &czero, u.Off(0, nr+1-1), ldu)
					Zlaset('A', toPtr((*m)-nr), toPtr(n1-nr), &czero, &cone, u.Off(nr+1-1, nr+1-1), ldu)
				}
			}

			Zunmqr('L', 'N', m, &n1, n, a, lda, cwork, u, ldu, cwork.Off((*n)+1-1), toPtr((*lwork)-(*n)), &ierr)

			if rowpiv {
				Zlaswp(&n1, u, ldu, func() *int { y := 1; return &y }(), toPtr((*m)-1), toSlice(iwork, iwoff+1-1), toPtr(-1))
			}

		}
		if transp {
			//           .. swap U and V because the procedure worked on A^*
			for p = 1; p <= (*n); p++ {
				goblas.Zswap(n, u.CVector(0, p-1), func() *int { y := 1; return &y }(), v.CVector(0, p-1), func() *int { y := 1; return &y }())
			}
		}

	}
	//     end of the full SVD
	//
	//     Undo scaling, if necessary (and possible)
	if uscal2 <= (big/sva.Get(0))*uscal1 {
		Dlascl('G', func() *int { y := 0; return &y }(), func() *int { y := 0; return &y }(), &uscal1, &uscal2, &nr, func() *int { y := 1; return &y }(), sva.Matrix(*n, opts), n, &ierr)
		uscal1 = one
		uscal2 = one
	}

	if nr < (*n) {
		for p = nr + 1; p <= (*n); p++ {
			sva.Set(p-1, zero)
		}
	}

	rwork.Set(0, uscal2*scalem)
	rwork.Set(1, uscal1)
	if errest {
		rwork.Set(2, sconda)
	}
	if lsvec && rsvec {
		rwork.Set(3, condr1)
		rwork.Set(4, condr2)
	}
	if l2tran {
		rwork.Set(5, entra)
		rwork.Set(6, entrat)
	}

	(*iwork)[0] = nr
	(*iwork)[1] = numrank
	(*iwork)[2] = warning
	if transp {
		(*iwork)[3] = 1
	} else {
		(*iwork)[3] = -1
	}
}
