package golapack

import (
	"golinalg/golapack/gltest"
	"golinalg/mat"
)

// Zuncsd computes the CS decomposition of an M-by-M partitioned
// unitary matrix X:
//
//                                 [  I  0  0 |  0  0  0 ]
//                                 [  0  C  0 |  0 -S  0 ]
//     [ X11 | X12 ]   [ U1 |    ] [  0  0  0 |  0  0 -I ] [ V1 |    ]**H
// X = [-----------] = [---------] [---------------------] [---------]   .
//     [ X21 | X22 ]   [    | U2 ] [  0  0  0 |  I  0  0 ] [    | V2 ]
//                                 [  0  S  0 |  0  C  0 ]
//                                 [  0  0  I |  0  0  0 ]
//
// X11 is P-by-Q. The unitary matrices U1, U2, V1, and V2 are P-by-P,
// (M-P)-by-(M-P), Q-by-Q, and (M-Q)-by-(M-Q), respectively. C and S are
// R-by-R nonnegative diagonal matrices satisfying C^2 + S^2 = I, in
// which R = minint(P,M-P,Q,M-Q).
func Zuncsd(jobu1, jobu2, jobv1t, jobv2t, trans, signs byte, m, p, q *int, x11 *mat.CMatrix, ldx11 *int, x12 *mat.CMatrix, ldx12 *int, x21 *mat.CMatrix, ldx21 *int, x22 *mat.CMatrix, ldx22 *int, theta *mat.Vector, u1 *mat.CMatrix, ldu1 *int, u2 *mat.CMatrix, ldu2 *int, v1t *mat.CMatrix, ldv1t *int, v2t *mat.CMatrix, ldv2t *int, work *mat.CVector, lwork *int, rwork *mat.Vector, lrwork *int, iwork *[]int, info *int) {
	var colmajor, defaultsigns, lquery, lrquery, wantu1, wantu2, wantv1t, wantv2t bool
	var signst, transt byte
	var one, zero complex128
	var childinfo, i, ib11d, ib11e, ib12d, ib12e, ib21d, ib21e, ib22d, ib22e, ibbcsd, iorbdb, iorglq, iorgqr, iphi, itaup1, itaup2, itauq1, itauq2, j, lbbcsdwork, lbbcsdworkmin, lbbcsdworkopt, lorbdbwork, lorbdbworkmin, lorbdbworkopt, lorglqwork, lorglqworkmin, lorglqworkopt, lorgqrwork, lorgqrworkmin, lorgqrworkopt, lrworkmin, lrworkopt, lworkmin, lworkopt, p1, q1 int

	one = (1.0 + 0.0*1i)
	zero = (0.0 + 0.0*1i)

	//     Test input arguments
	(*info) = 0
	wantu1 = jobu1 == 'Y'
	wantu2 = jobu2 == 'Y'
	wantv1t = jobv1t == 'Y'
	wantv2t = jobv2t == 'Y'
	colmajor = trans != 'T'
	defaultsigns = signs != 'O'
	lquery = (*lwork) == -1
	lrquery = (*lrwork) == -1
	if (*m) < 0 {
		(*info) = -7
	} else if (*p) < 0 || (*p) > (*m) {
		(*info) = -8
	} else if (*q) < 0 || (*q) > (*m) {
		(*info) = -9
	} else if colmajor && (*ldx11) < maxint(1, *p) {
		(*info) = -11
	} else if !colmajor && (*ldx11) < maxint(1, *q) {
		(*info) = -11
	} else if colmajor && (*ldx12) < maxint(1, *p) {
		(*info) = -13
	} else if !colmajor && (*ldx12) < maxint(1, (*m)-(*q)) {
		(*info) = -13
	} else if colmajor && (*ldx21) < maxint(1, (*m)-(*p)) {
		(*info) = -15
	} else if !colmajor && (*ldx21) < maxint(1, *q) {
		(*info) = -15
	} else if colmajor && (*ldx22) < maxint(1, (*m)-(*p)) {
		(*info) = -17
	} else if !colmajor && (*ldx22) < maxint(1, (*m)-(*q)) {
		(*info) = -17
	} else if wantu1 && (*ldu1) < (*p) {
		(*info) = -20
	} else if wantu2 && (*ldu2) < (*m)-(*p) {
		(*info) = -22
	} else if wantv1t && (*ldv1t) < (*q) {
		(*info) = -24
	} else if wantv2t && (*ldv2t) < (*m)-(*q) {
		(*info) = -26
	}

	//     Work with transpose if convenient
	if (*info) == 0 && minint(*p, (*m)-(*p)) < minint(*q, (*m)-(*q)) {
		if colmajor {
			transt = 'T'
		} else {
			transt = 'N'
		}
		if defaultsigns {
			signst = 'O'
		} else {
			signst = 'D'
		}
		Zuncsd(jobv1t, jobv2t, jobu1, jobu2, transt, signst, m, q, p, x11, ldx11, x21, ldx21, x12, ldx12, x22, ldx22, theta, v1t, ldv1t, v2t, ldv2t, u1, ldu1, u2, ldu2, work, lwork, rwork, lrwork, iwork, info)
		return
	}

	//     Work with permutation [ 0 I; I 0 ] * X * [ 0 I; I 0 ] if
	//     convenient
	if (*info) == 0 && (*m)-(*q) < (*q) {
		if defaultsigns {
			signst = 'O'
		} else {
			signst = 'D'
		}
		Zuncsd(jobu2, jobu1, jobv2t, jobv1t, trans, signst, m, toPtr((*m)-(*p)), toPtr((*m)-(*q)), x22, ldx22, x21, ldx21, x12, ldx12, x11, ldx11, theta, u2, ldu2, u1, ldu1, v2t, ldv2t, v1t, ldv1t, work, lwork, rwork, lrwork, iwork, info)
		return
	}

	//     Compute workspace
	if (*info) == 0 {
		//        Real workspace
		iphi = 2
		ib11d = iphi + maxint(1, (*q)-1)
		ib11e = ib11d + maxint(1, *q)
		ib12d = ib11e + maxint(1, (*q)-1)
		ib12e = ib12d + maxint(1, *q)
		ib21d = ib12e + maxint(1, (*q)-1)
		ib21e = ib21d + maxint(1, *q)
		ib22d = ib21e + maxint(1, (*q)-1)
		ib22e = ib22d + maxint(1, *q)
		ibbcsd = ib22e + maxint(1, (*q)-1)
		Zbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q, theta, theta, u1, ldu1, u2, ldu2, v1t, ldv1t, v2t, ldv2t, theta, theta, theta, theta, theta, theta, theta, theta, rwork, toPtr(-1), &childinfo)
		lbbcsdworkopt = int(rwork.Get(0))
		lbbcsdworkmin = lbbcsdworkopt
		lrworkopt = ibbcsd + lbbcsdworkopt - 1
		lrworkmin = ibbcsd + lbbcsdworkmin - 1
		rwork.Set(0, float64(lrworkopt))

		//        Complex workspace
		itaup1 = 2
		itaup2 = itaup1 + maxint(1, *p)
		itauq1 = itaup2 + maxint(1, (*m)-(*p))
		itauq2 = itauq1 + maxint(1, *q)
		iorgqr = itauq2 + maxint(1, (*m)-(*q))
		Zungqr(toPtr((*m)-(*q)), toPtr((*m)-(*q)), toPtr((*m)-(*q)), u1, toPtr(maxint(1, (*m)-(*q))), u1.CVector(0, 0), work, toPtr(-1), &childinfo)
		lorgqrworkopt = int(work.GetRe(0))
		lorgqrworkmin = maxint(1, (*m)-(*q))
		iorglq = itauq2 + maxint(1, (*m)-(*q))
		Zunglq(toPtr((*m)-(*q)), toPtr((*m)-(*q)), toPtr((*m)-(*q)), u1, toPtr(maxint(1, (*m)-(*q))), u1.CVector(0, 0), work, toPtr(-1), &childinfo)
		lorglqworkopt = int(work.GetRe(0))
		lorglqworkmin = maxint(1, (*m)-(*q))
		iorbdb = itauq2 + maxint(1, (*m)-(*q))
		Zunbdb(trans, signs, m, p, q, x11, ldx11, x12, ldx12, x21, ldx21, x22, ldx22, theta, theta, u1.CVector(0, 0), u2.CVector(0, 0), v1t.CVector(0, 0), v2t.CVector(0, 0), work, toPtr(-1), &childinfo)
		lorbdbworkopt = int(work.GetRe(0))
		lorbdbworkmin = lorbdbworkopt
		lworkopt = maxint(iorgqr+lorgqrworkopt, iorglq+lorglqworkopt, iorbdb+lorbdbworkopt) - 1
		lworkmin = maxint(iorgqr+lorgqrworkmin, iorglq+lorglqworkmin, iorbdb+lorbdbworkmin) - 1
		work.SetRe(0, float64(maxint(lworkopt, lworkmin)))

		if (*lwork) < lworkmin && !(lquery || lrquery) {
			(*info) = -22
		} else if (*lrwork) < lrworkmin && !(lquery || lrquery) {
			(*info) = -24
		} else {
			lorgqrwork = (*lwork) - iorgqr + 1
			lorglqwork = (*lwork) - iorglq + 1
			lorbdbwork = (*lwork) - iorbdb + 1
			lbbcsdwork = (*lrwork) - ibbcsd + 1
		}
	}

	//     Abort if any illegal arguments
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZUNCSD"), -(*info))
		return
	} else if lquery || lrquery {
		return
	}

	//     Transform to bidiagonal block form
	Zunbdb(trans, signs, m, p, q, x11, ldx11, x12, ldx12, x21, ldx21, x22, ldx22, theta, rwork.Off(iphi-1), work.Off(itaup1-1), work.Off(itaup2-1), work.Off(itauq1-1), work.Off(itauq2-1), work.Off(iorbdb-1), &lorbdbwork, &childinfo)

	//     Accumulate Householder reflectors
	if colmajor {
		if wantu1 && (*p) > 0 {
			Zlacpy('L', p, q, x11, ldx11, u1, ldu1)
			Zungqr(p, p, q, u1, ldu1, work.Off(itaup1-1), work.Off(iorgqr-1), &lorgqrwork, info)
		}
		if wantu2 && (*m)-(*p) > 0 {
			Zlacpy('L', toPtr((*m)-(*p)), q, x21, ldx21, u2, ldu2)
			Zungqr(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, work.Off(itaup2-1), work.Off(iorgqr-1), &lorgqrwork, info)
		}
		if wantv1t && (*q) > 0 {
			Zlacpy('U', toPtr((*q)-1), toPtr((*q)-1), x11.Off(0, 1), ldx11, v1t.Off(1, 1), ldv1t)
			v1t.Set(0, 0, one)
			for j = 2; j <= (*q); j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			Zunglq(toPtr((*q)-1), toPtr((*q)-1), toPtr((*q)-1), v1t.Off(1, 1), ldv1t, work.Off(itauq1-1), work.Off(iorglq-1), &lorglqwork, info)
		}
		if wantv2t && (*m)-(*q) > 0 {
			Zlacpy('U', p, toPtr((*m)-(*q)), x12, ldx12, v2t, ldv2t)
			if (*m)-(*p) > (*q) {
				Zlacpy('U', toPtr((*m)-(*p)-(*q)), toPtr((*m)-(*p)-(*q)), x22.Off((*q)+1-1, (*p)+1-1), ldx22, v2t.Off((*p)+1-1, (*p)+1-1), ldv2t)
			}
			if (*m) > (*q) {
				Zunglq(toPtr((*m)-(*q)), toPtr((*m)-(*q)), toPtr((*m)-(*q)), v2t, ldv2t, work.Off(itauq2-1), work.Off(iorglq-1), &lorglqwork, info)
			}
		}
	} else {
		if wantu1 && (*p) > 0 {
			Zlacpy('U', q, p, x11, ldx11, u1, ldu1)
			Zunglq(p, p, q, u1, ldu1, work.Off(itaup1-1), work.Off(iorglq-1), &lorglqwork, info)
		}
		if wantu2 && (*m)-(*p) > 0 {
			Zlacpy('U', q, toPtr((*m)-(*p)), x21, ldx21, u2, ldu2)
			Zunglq(toPtr((*m)-(*p)), toPtr((*m)-(*p)), q, u2, ldu2, work.Off(itaup2-1), work.Off(iorglq-1), &lorglqwork, info)
		}
		if wantv1t && (*q) > 0 {
			Zlacpy('L', toPtr((*q)-1), toPtr((*q)-1), x11.Off(1, 0), ldx11, v1t.Off(1, 1), ldv1t)
			v1t.Set(0, 0, one)
			for j = 2; j <= (*q); j++ {
				v1t.Set(0, j-1, zero)
				v1t.Set(j-1, 0, zero)
			}
			Zungqr(toPtr((*q)-1), toPtr((*q)-1), toPtr((*q)-1), v1t.Off(1, 1), ldv1t, work.Off(itauq1-1), work.Off(iorgqr-1), &lorgqrwork, info)
		}
		if wantv2t && (*m)-(*q) > 0 {
			p1 = minint((*p)+1, *m)
			q1 = minint((*q)+1, *m)
			Zlacpy('L', toPtr((*m)-(*q)), p, x12, ldx12, v2t, ldv2t)
			if (*m) > (*p)+(*q) {
				Zlacpy('L', toPtr((*m)-(*p)-(*q)), toPtr((*m)-(*p)-(*q)), x22.Off(p1-1, q1-1), ldx22, v2t.Off((*p)+1-1, (*p)+1-1), ldv2t)
			}
			Zungqr(toPtr((*m)-(*q)), toPtr((*m)-(*q)), toPtr((*m)-(*q)), v2t, ldv2t, work.Off(itauq2-1), work.Off(iorgqr-1), &lorgqrwork, info)
		}
	}

	//     Compute the CSD of the matrix in bidiagonal-block form
	Zbbcsd(jobu1, jobu2, jobv1t, jobv2t, trans, m, p, q, theta, rwork.Off(iphi-1), u1, ldu1, u2, ldu2, v1t, ldv1t, v2t, ldv2t, rwork.Off(ib11d-1), rwork.Off(ib11e-1), rwork.Off(ib12d-1), rwork.Off(ib12e-1), rwork.Off(ib21d-1), rwork.Off(ib21e-1), rwork.Off(ib22d-1), rwork.Off(ib22e-1), rwork.Off(ibbcsd-1), &lbbcsdwork, info)

	//     Permute rows and columns to place identity submatrices in top-
	//     left corner of (1,1)-block and/or bottom-right corner of (1,2)-
	//     block and/or bottom-right corner of (2,1)-block and/or top-left
	//     corner of (2,2)-block
	if (*q) > 0 && wantu2 {
		for i = 1; i <= (*q); i++ {
			(*iwork)[i-1] = (*m) - (*p) - (*q) + i
		}
		for i = (*q) + 1; i <= (*m)-(*p); i++ {
			(*iwork)[i-1] = i - (*q)
		}
		if colmajor {
			Zlapmt(false, toPtr((*m)-(*p)), toPtr((*m)-(*p)), u2, ldu2, iwork)
		} else {
			Zlapmr(false, toPtr((*m)-(*p)), toPtr((*m)-(*p)), u2, ldu2, iwork)
		}
	}
	if (*m) > 0 && wantv2t {
		for i = 1; i <= (*p); i++ {
			(*iwork)[i-1] = (*m) - (*p) - (*q) + i
		}
		for i = (*p) + 1; i <= (*m)-(*q); i++ {
			(*iwork)[i-1] = i - (*p)
		}
		if !colmajor {
			Zlapmt(false, toPtr((*m)-(*q)), toPtr((*m)-(*q)), v2t, ldv2t, iwork)
		} else {
			Zlapmr(false, toPtr((*m)-(*q)), toPtr((*m)-(*q)), v2t, ldv2t, iwork)
		}
	}
}
