package golapack

import (
	"fmt"
	"math"

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
func Dgesvdq(joba, jobp, jobr, jobu, jobv byte, m, n int, a *mat.Matrix, s *mat.Vector, u, v *mat.Matrix, iwork *[]int, liwork int, work *mat.Vector, lwork int, rwork *mat.Vector, lrwork int) (numrank, info int, err error) {
	var accla, acclh, acclm, ascaled, conda, dntwu, dntwv, lquery, lsvc0, lsvec, rowprm, rsvec, rtrans, wntua, wntuf, wntur, wntus, wntva, wntvr bool
	var big, epsln, one, rtmp, sconda, sfmin, zero float64
	var iminwrk, iwoff, lwcon, lwlqf, lworlq, lworq, lworq2, lwqp3, lwqrf, lwrkDgelqf, lwrkDgeqp3, lwrkDgeqrf, lwrkDgesvd, lwrkDgesvd2, lwrkDormlq, lwrkDormqr, lwrkDormqr2, lwsvd, lwsvd2, minwrk, minwrk2, n1, nr, optratio, optwrk, optwrk2, p, q, rminwrk int

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
			iminwrk = max(1, n+m-1+n)
		} else {
			iminwrk = max(1, n+m-1)
		}
		rminwrk = max(2, m)
	} else {
		if conda {
			iminwrk = max(1, n+n)
		} else {
			iminwrk = max(1, n)
		}
		rminwrk = 2
	}
	lquery = (liwork == -1 || lwork == -1 || lrwork == -1)
	if !(accla || acclm || acclh) {
		err = fmt.Errorf("!(accla || acclm || acclh): joba='%c'", joba)
	} else if !(rowprm || jobp == 'N') {
		err = fmt.Errorf("!(rowprm || jobp == 'N'): jobp='%c'", jobp)
	} else if !(rtrans || jobr == 'N') {
		err = fmt.Errorf("!(rtrans || jobr == 'N'): jobr='%c'", jobr)
	} else if !(lsvec || dntwu) {
		err = fmt.Errorf("!(lsvec || dntwu): jobu='%c'", jobu)
	} else if wntur && wntva {
		err = fmt.Errorf("wntur && wntva: jobv='%c'", jobv)
	} else if !(rsvec || dntwv) {
		err = fmt.Errorf("!(rsvec || dntwv): jobv='%c'", jobv)
	} else if m < 0 {
		err = fmt.Errorf("m < 0: m=%v", m)
	} else if (n < 0) || (n > m) {
		err = fmt.Errorf("(n < 0) || (n > m): m=%v, n=%v", m, n)
	} else if a.Rows < max(1, m) {
		err = fmt.Errorf("a.Rows < max(1, m): a.Rows=%v, m=%v", a.Rows, m)
	} else if u.Rows < 1 || (lsvc0 && u.Rows < m) || (wntuf && u.Rows < n) {
		err = fmt.Errorf("u.Rows < 1 || (lsvc0 && u.Rows < m) || (wntuf && u.Rows < n): jobu='%c', u.Rows=%v, m=%v, n=%v", jobu, u.Rows, m, n)
	} else if v.Rows < 1 || (rsvec && v.Rows < n) || (conda && v.Rows < n) {
		err = fmt.Errorf("v.Rows < 1 || (rsvec && v.Rows < n) || (conda && v.Rows < n): jobv='%c', v.Rows=%v, n=%v", jobv, v.Rows, n)
	} else if liwork < iminwrk && !lquery {
		err = fmt.Errorf("liwork < iminwrk && !lquery: liwork=%v, iminwrk=%v, lquery=%v", liwork, iminwrk, lquery)
	}

	if err == nil {
		//        .. compute the minimal and the optimal workspace lengths
		//        [[The expressions for computing the minimal and the optimal
		//        values of LWORK are written with a lot of redundancy and
		//        can be simplified. However, this detailed form is easier for
		//        maintenance and modifications of the code.]]
		//
		//        .. minimal workspace length for DGEQP3 of an M x N matrix
		lwqp3 = 3*n + 1
		//        .. minimal workspace length for DORMQR to build left singular vectors
		if wntus || wntur {
			lworq = max(n, 1)
		} else if wntua {
			lworq = max(m, 1)
		}
		//        .. minimal workspace length for DPOCON of an N x N matrix
		lwcon = 3 * n
		//        .. DGESVD of an N x N matrix
		lwsvd = max(5*n, 1)
		if lquery {
			if err = Dgeqp3(m, n, a, iwork, rdummy, rdummy, -1); err != nil {
				panic(err)
			}
			lwrkDgeqp3 = int(rdummy.Get(0))
			if wntus || wntur {
				if err = Dormqr(Left, NoTrans, m, n, n, a, rdummy, u, rdummy, -1); err != nil {
					panic(err)
				}
				lwrkDormqr = int(rdummy.Get(0))
			} else if wntua {
				if err = Dormqr(Left, NoTrans, m, m, n, a, rdummy, u, rdummy, -1); err != nil {
					panic(err)
				}
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
				minwrk = max(n+lwqp3, lwcon, lwsvd)
			} else {
				minwrk = max(n+lwqp3, lwsvd)
			}
			if lquery {
				if _, err = Dgesvd('N', 'N', n, n, a, s, u, v, rdummy, -1); err != nil {
					panic(err)
				}
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = max(n+lwrkDgeqp3, n+lwcon, lwrkDgesvd)
				} else {
					optwrk = max(n+lwrkDgeqp3, lwrkDgesvd)
				}
			}
		} else if lsvec && (!rsvec) {
			//            .. minimal and optimal sizes of the workspace if the
			//            singular values and the left singular vectors are requested
			if conda {
				minwrk = n + max(lwqp3, lwcon, lwsvd, lworq)
			} else {
				minwrk = n + max(lwqp3, lwsvd, lworq)
			}
			if lquery {
				if rtrans {
					if _, err = Dgesvd('N', 'O', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
				} else {
					if _, err = Dgesvd('O', 'N', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
				}
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = n + max(lwrkDgeqp3, lwcon, lwrkDgesvd, lwrkDormqr)
				} else {
					optwrk = n + max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
				}
			}
		} else if rsvec && (!lsvec) {
			//            .. minimal and optimal sizes of the workspace if the
			//            singular values and the right singular vectors are requested
			if conda {
				minwrk = n + max(lwqp3, lwcon, lwsvd)
			} else {
				minwrk = n + max(lwqp3, lwsvd)
			}
			if lquery {
				if rtrans {
					if _, err = Dgesvd('O', 'N', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
				} else {
					if _, err = Dgesvd('N', 'O', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
				}
				lwrkDgesvd = int(rdummy.Get(0))
				if conda {
					optwrk = n + max(lwrkDgeqp3, lwcon, lwrkDgesvd)
				} else {
					optwrk = n + max(lwrkDgeqp3, lwrkDgesvd)
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
				minwrk = minwrk + n
				if wntva {
					//                   .. minimal workspace length for N x N/2 DGEQRF
					lwqrf = max(n/2, 1)
					//                   .. minimal workspace lengt for N/2 x N/2 DGESVD
					lwsvd2 = max(5*(n/2), 1)
					lworq2 = max(n, 1)
					minwrk2 = max(lwqp3, n/2+lwqrf, n/2+lwsvd2, n/2+lworq2, lworq)
					if conda {
						minwrk2 = max(minwrk2, lwcon)
					}
					minwrk2 = n + minwrk2
					minwrk = max(minwrk, minwrk2)
				}
			} else {
				minwrk = max(lwqp3, lwsvd, lworq)
				if conda {
					minwrk = max(minwrk, lwcon)
				}
				minwrk = minwrk + n
				if wntva {
					//                   .. minimal workspace length for N/2 x N DGELQF
					lwlqf = max(n/2, 1)
					lwsvd2 = max(5*(n/2), 1)
					lworlq = max(n, 1)
					minwrk2 = max(lwqp3, n/2+lwlqf, n/2+lwsvd2, n/2+lworlq, lworq)
					if conda {
						minwrk2 = max(minwrk2, lwcon)
					}
					minwrk2 = n + minwrk2
					minwrk = max(minwrk, minwrk2)
				}
			}
			if lquery {
				if rtrans {
					if _, err = Dgesvd('O', 'A', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
					lwrkDgesvd = int(rdummy.Get(0))
					optwrk = max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
					if conda {
						optwrk = max(optwrk, lwcon)
					}
					optwrk = n + optwrk
					if wntva {
						if err = Dgeqrf(n, n/2, u, rdummy, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDgeqrf = int(rdummy.Get(0))
						if _, err = Dgesvd('S', 'O', n/2, n/2, v, s, u, v, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDgesvd2 = int(rdummy.Get(0))
						if err = Dormqr(Right, ConjTrans, n, n, n/2, u, rdummy, v, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDormqr2 = int(rdummy.Get(0))
						optwrk2 = max(lwrkDgeqp3, n/2+lwrkDgeqrf, n/2+lwrkDgesvd2, n/2+lwrkDormqr2)
						if conda {
							optwrk2 = max(optwrk2, lwcon)
						}
						optwrk2 = n + optwrk2
						optwrk = max(optwrk, optwrk2)
					}
				} else {
					if _, err = Dgesvd('S', 'O', n, n, a, s, u, v, rdummy, -1); err != nil {
						panic(err)
					}
					lwrkDgesvd = int(rdummy.Get(0))
					optwrk = max(lwrkDgeqp3, lwrkDgesvd, lwrkDormqr)
					if conda {
						optwrk = max(optwrk, lwcon)
					}
					optwrk = n + optwrk
					if wntva {
						if err = Dgelqf(n/2, n, u, rdummy, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDgelqf = int(rdummy.Get(0))
						if _, err = Dgesvd('S', 'O', n/2, n/2, v, s, u, v, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDgesvd2 = int(rdummy.Get(0))
						if err = Dormlq(Right, NoTrans, n, n, n/2, u, rdummy, v, rdummy, -1); err != nil {
							panic(err)
						}
						lwrkDormlq = int(rdummy.Get(0))
						optwrk2 = max(lwrkDgeqp3, n/2+lwrkDgelqf, n/2+lwrkDgesvd2, n/2+lwrkDormlq)
						if conda {
							optwrk2 = max(optwrk2, lwcon)
						}
						optwrk2 = n + optwrk2
						optwrk = max(optwrk, optwrk2)
					}
				}
			}
		}

		minwrk = max(2, minwrk)
		optwrk = max(2, optwrk)
		if lwork < minwrk && (!lquery) {
			err = fmt.Errorf("lwork < minwrk && (!lquery): lwork=%v, minwrk=%v, lquery=%v", lwork, minwrk, lquery)
		}

	}

	if err == nil && lrwork < rminwrk && !lquery {
		err = fmt.Errorf("lrwork < rminwrk && !lquery: lrwork=%v, rminwrk=%v, lquery=%v", lrwork, rminwrk, lquery)
	}
	if err != nil {
		gltest.Xerbla2("Dgesvdq", err)
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
	if (m == 0) || (n == 0) {
		//     .. all output is void.
		return
	}

	big = Dlamch(Overflow)
	ascaled = false
	iwoff = 1
	if rowprm {
		iwoff = m
		//           .. reordering the rows in decreasing sequence in the
		//           ell-infinity norm - this enhances numerical robustness in
		//           the case of differently scaled rows.
		for p = 1; p <= m; p++ {
			//               RWORK(p) = ABS( A(p,ICAMAX(N,A(p,1),LDA)) )
			//               [[DLANGE will return NaN if an entry of the p-th row is Nan]]
			rwork.Set(p-1, Dlange('M', 1, n, a.Off(p-1, 0), rdummy))
			//               .. check for NaN's and Inf's
			if (rwork.Get(p-1) != rwork.Get(p-1)) || ((rwork.Get(p-1) * zero) != zero) {
				err = fmt.Errorf("(rwork.Get(p-1) != rwork.Get(p-1)) || ((rwork.Get(p-1) * zero) != zero): rwork(p-1)=%v, p=%v", rwork.Get(p-1), p)
				gltest.Xerbla2("Dgesvdq", err)
				return
			}
		}
		for p = 1; p <= m-1; p++ {
			q = rwork.Off(p-1).Iamax(m-p+1, 1) + p - 1
			(*iwork)[n+p-1] = q
			if p != q {
				rtmp = rwork.Get(p - 1)
				rwork.Set(p-1, rwork.Get(q-1))
				rwork.Set(q-1, rtmp)
			}
		}

		if rwork.Get(0) == zero {
			//              Quick return: A is the M x N zero matrix.
			numrank = 0
			Dlaset(Full, n, 1, zero, zero, s.Matrix(n, opts))
			if wntus {
				Dlaset(Full, m, n, zero, one, u)
			}
			if wntua {
				Dlaset(Full, m, m, zero, one, u)
			}
			if wntva {
				Dlaset(Full, n, n, zero, one, v)
			}
			if wntuf {
				Dlaset(Full, n, 1, zero, zero, work.Matrix(n, opts))
				Dlaset(Full, m, n, zero, one, u)
			}
			for p = 1; p <= n; p++ {
				(*iwork)[p-1] = p
			}
			if rowprm {
				for p = n + 1; p <= n+m-1; p++ {
					(*iwork)[p-1] = p - n
				}
			}
			if conda {
				rwork.Set(0, -1)
			}
			rwork.Set(1, -1)
			return
		}

		if rwork.Get(0) > big/math.Sqrt(float64(m)) {
			//               .. to prevent overflow in the QR factorization, scale the
			//               matrix by 1/math.Sqrt(M) if too large entry detected
			if err = Dlascl('G', 0, 0, math.Sqrt(float64(m)), one, m, n, a); err != nil {
				panic(err)
			}
			ascaled = true
		}
		Dlaswp(n, a, 1, m-1, *toSlice(iwork, n), 1)
	}

	//    .. At this stage, preemptive scaling is done only to avoid column
	//    norms overflows during the QR factorization. The SVD procedure should
	//    have its own scaling to save the singular values from overflows and
	//    underflows. That depends on the SVD procedure.
	if !rowprm {
		rtmp = Dlange('M', m, n, a, rdummy)
		if (rtmp != rtmp) || ((rtmp * zero) != zero) {
			err = fmt.Errorf("(rtmp != rtmp) || ((rtmp * zero) != zero): rtmp=%v", rtmp)
			gltest.Xerbla2("Dgesvdq", err)
			return
		}
		if rtmp > big/math.Sqrt(float64(m)) {
			//             .. to prevent overflow in the QR factorization, scale the
			//             matrix by 1/math.Sqrt(M) if too large entry detected
			if err = Dlascl('G', 0, 0, math.Sqrt(float64(m)), one, m, n, a); err != nil {
				panic(err)
			}
			ascaled = true
		}
	}

	//     .. QR factorization with column pivoting
	//
	//     A * P = Q * [ R ]
	//                 [ 0 ]
	for p = 1; p <= n; p++ {
		//        .. all columns are free columns
		(*iwork)[p-1] = 0
	}
	if err = Dgeqp3(m, n, a, iwork, work, work.Off(n), lwork-n); err != nil {
		panic(err)
	}

	//    If the user requested accuracy level allows truncation in the
	//    computed upper triangular factor, the matrix R is examined and,
	//    if possible, replaced with its leading upper trapezoidal part.
	epsln = Dlamch(Epsilon)
	sfmin = Dlamch(SafeMinimum)
	//     SMALL = SFMIN / EPSLN
	nr = n

	if accla {
		//        Standard absolute error bound suffices. All sigma_i with
		//        sigma_i < N*EPS*||A||_F are flushed to zero. This is an
		//        aggressive enforcement of lower numerical rank by introducing a
		//        backward error of the order of N*EPS*||A||_F.
		nr = 1
		rtmp = math.Sqrt(float64(n)) * epsln
		for p = 2; p <= n; p++ {
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
		for p = 2; p <= n; p++ {
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
		for p = 2; p <= n; p++ {
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
			Dlacpy(Upper, n, n, a, v)
			//              Only the leading NR x NR submatrix of the triangular factor
			//              is considered. Only if NR=N will this give a reliable error
			//              bound. However, even for NR < N, this can be used on an
			//              expert level and obtain useful information in the sense of
			//              perturbation theory.
			for p = 1; p <= nr; p++ {
				rtmp = v.Off(0, p-1).Vector().Nrm2(p, 1)
				v.Off(0, p-1).Vector().Scal(p, one/rtmp, 1)
			}
			if !(lsvec || rsvec) {
				if rtmp, err = Dpocon(Upper, nr, v, one, work, toSlice(iwork, n+iwoff-1)); err != nil {
					panic(err)
				}
			} else {
				if rtmp, err = Dpocon(Upper, nr, v, one, work.Off(n), toSlice(iwork, n+iwoff-1)); err != nil {
					panic(err)
				}
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
		n1 = n
	} else if wntua {
		n1 = m
	}

	if !(rsvec || lsvec) {
		//.......................................................................
		//        .. only the singular values are requested
		//.......................................................................
		if rtrans {
			//         .. compute the singular values of R**T = [A](1:NR,1:N)**T
			//           .. set the lower triangle of [A] to [A](1:NR,1:N)**T and
			//           the upper triangle of [A] to zero.
			for p = 1; p <= min(n, nr); p++ {
				for q = p + 1; q <= n; q++ {
					a.Set(q-1, p-1, a.Get(p-1, q-1))
					if q <= nr {
						a.Set(p-1, q-1, zero)
					}
				}
			}

			if info, err = Dgesvd('N', 'N', n, nr, a, s, u, v, work, lwork); err != nil {
				panic(err)
			}

		} else {
			//           .. compute the singular values of R = [A](1:NR,1:N)
			if nr > 1 {
				Dlaset(Lower, nr-1, nr-1, zero, zero, a.Off(1, 0))
			}
			if info, err = Dgesvd('N', 'N', nr, n, a, s, u, v, work, lwork); err != nil {
				panic(err)
			}

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
				for q = p; q <= n; q++ {
					u.Set(q-1, p-1, a.Get(p-1, q-1))
				}
			}
			if nr > 1 {
				Dlaset(Upper, nr-1, nr-1, zero, zero, u.Off(0, 1))
			}
			//           .. the left singular vectors not computed, the NR right singular
			//           vectors overwrite [U](1:NR,1:NR) as transposed. These
			//           will be pre-multiplied by Q to build the left singular vectors of A.
			if info, err = Dgesvd('N', 'O', n, nr, u, s, u, u, work.Off(n), lwork-n); err != nil {
				panic(err)
			}

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
			Dlacpy(Upper, nr, n, a, u)
			if nr > 1 {
				Dlaset(Lower, nr-1, nr-1, zero, zero, u.Off(1, 0))
			}
			//            .. the right singular vectors not computed, the NR left singular
			//            vectors overwrite [U](1:NR,1:NR)
			if info, err = Dgesvd('O', 'N', nr, n, u, s, u, v, work.Off(n), lwork-n); err != nil {
				panic(err)
			}
			//               .. now [U](1:NR,1:NR) contains the NR left singular vectors of
			//               R. These will be pre-multiplied by Q to build the left singular
			//               vectors of A.
		}

		//           .. assemble the left singular vector matrix U of dimensions
		//              (M x NR) or (M x N) or (M x M).
		if (nr < m) && (!wntuf) {
			Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
			if nr < n1 {
				Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
				Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
			}
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
				panic(err)
			}
		}
		if rowprm && !wntuf {
			Dlaswp(n1, u, 1, m-1, *toSlice(iwork, n), -1)
		}

	} else if rsvec && (!lsvec) {
		//.......................................................................
		//       .. the singular values and the right singular vectors requested
		//.......................................................................
		if rtrans {
			//            .. apply DGESVD to R**T
			//            .. copy R**T into V and overwrite V with the left singular vectors
			for p = 1; p <= nr; p++ {
				for q = p; q <= n; q++ {
					v.Set(q-1, p-1, (a.Get(p-1, q-1)))
				}
			}
			if nr > 1 {
				Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
			}
			//           .. the left singular vectors of R**T overwrite V, the right singular
			//           vectors not computed
			if wntvr || (nr == n) {
				if info, err = Dgesvd('O', 'N', n, nr, v, s, u, u, work.Off(n), lwork-n); err != nil {
					panic(err)
				}

				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}

				if nr < n {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= n; q++ {
							v.Set(p-1, q-1, v.Get(q-1, p-1))
						}
					}
				}
				Dlapmt(false, nr, n, v, iwork)
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:N,1:NR)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the QR factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Dlaset(Full, n, n-nr, zero, zero, v.Off(0, nr))
				if info, err = Dgesvd('O', 'N', n, n, v, s, u, u, work.Off(n), lwork-n); err != nil {
					panic(err)
				}

				for p = 1; p <= n; p++ {
					for q = p + 1; q <= n; q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}
				Dlapmt(false, n, n, v, iwork)
			}

		} else {
			//            .. aply DGESVD to R
			//            .. copy R into V and overwrite V with the right singular vectors
			Dlacpy(Upper, nr, n, a, v)
			if nr > 1 {
				Dlaset(Lower, nr-1, nr-1, zero, zero, v.Off(1, 0))
			}
			//            .. the right singular vectors overwrite V, the NR left singular
			//            vectors stored in U(1:NR,1:NR)
			if wntvr || (nr == n) {
				if info, err = Dgesvd('N', 'O', nr, n, v, s, u, v, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				Dlapmt(false, nr, n, v, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**T
			} else {
				//               .. need all N right singular vectors and NR < N
				//               [!] This is simple implementation that augments [V](1:NR,1:N)
				//               by padding a zero block. In the case NR << N, a more efficient
				//               way is to first use the LQ factorization. For more details
				//               how to implement this, see the " FULL SVD " branch.
				Dlaset(Full, n-nr, n, zero, zero, v.Off(nr, 0))
				if info, err = Dgesvd('N', 'O', n, n, v, s, u, v, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				Dlapmt(false, n, n, v, iwork)
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
			if wntvr || (nr == n) {
				//            .. copy R**T into [V] and overwrite [V] with the left singular
				//            vectors of R**T
				for p = 1; p <= nr; p++ {
					for q = p; q <= n; q++ {
						v.Set(q-1, p-1, a.Get(p-1, q-1))
					}
				}
				if nr > 1 {
					Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
				}

				//           .. the left singular vectors of R**T overwrite [V], the NR right
				//           singular vectors of R**T stored in [U](1:NR,1:NR) as transposed
				if info, err = Dgesvd('O', 'A', n, nr, v, s, v, u, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				//              .. assemble V
				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = v.Get(q-1, p-1)
						v.Set(q-1, p-1, v.Get(p-1, q-1))
						v.Set(p-1, q-1, rtmp)
					}
				}
				if nr < n {
					for p = 1; p <= nr; p++ {
						for q = nr + 1; q <= n; q++ {
							v.Set(p-1, q-1, v.Get(q-1, p-1))
						}
					}
				}
				Dlapmt(false, nr, n, v, iwork)

				for p = 1; p <= nr; p++ {
					for q = p + 1; q <= nr; q++ {
						rtmp = u.Get(q-1, p-1)
						u.Set(q-1, p-1, u.Get(p-1, q-1))
						u.Set(p-1, q-1, rtmp)
					}
				}

				if (nr < m) && !wntuf {
					Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
					if nr < n1 {
						Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
						Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
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
				if optratio*nr > n {
					for p = 1; p <= nr; p++ {
						for q = p; q <= n; q++ {
							v.Set(q-1, p-1, a.Get(p-1, q-1))
						}
					}
					if nr > 1 {
						Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
					}

					Dlaset(Full, n, n-nr, zero, zero, v.Off(0, nr))
					if info, err = Dgesvd('O', 'A', n, n, v, s, v, u, work.Off(n), lwork-n); err != nil {
						panic(err)
					}

					for p = 1; p <= n; p++ {
						for q = p + 1; q <= n; q++ {
							rtmp = v.Get(q-1, p-1)
							v.Set(q-1, p-1, v.Get(p-1, q-1))
							v.Set(p-1, q-1, rtmp)
						}
					}
					Dlapmt(false, n, n, v, iwork)
					//              .. assemble the left singular vector matrix U of dimensions
					//              (M x N1), i.e. (M x N) or (M x M).

					for p = 1; p <= n; p++ {
						for q = p + 1; q <= n; q++ {
							rtmp = u.Get(q-1, p-1)
							u.Set(q-1, p-1, u.Get(p-1, q-1))
							u.Set(p-1, q-1, rtmp)
						}
					}

					if (n < m) && !wntuf {
						Dlaset(Full, m-n, n, zero, zero, u.Off(n, 0))
						if n < n1 {
							Dlaset(Full, n, n1-n, zero, zero, u.Off(0, n))
							Dlaset(Full, m-n, n1-n, zero, one, u.Off(n, n))
						}
					}
				} else {
					//                  .. copy R**T into [U] and overwrite [U] with the right
					//                  singular vectors of R
					for p = 1; p <= nr; p++ {
						for q = p; q <= n; q++ {
							u.Set(q-1, nr+p-1, a.Get(p-1, q-1))
						}
					}
					if nr > 1 {
						Dlaset(Upper, nr-1, nr-1, zero, zero, u.Off(0, nr+2-1))
					}
					if err = Dgeqrf(n, nr, u.Off(0, nr), work.Off(n), work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					for p = 1; p <= nr; p++ {
						for q = 1; q <= n; q++ {
							v.Set(q-1, p-1, u.Get(p-1, nr+q-1))
						}
					}
					Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
					if info, err = Dgesvd('S', 'O', nr, nr, v, s, u, v, work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
					Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
					Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
					if err = Dormqr(Right, ConjTrans, n, n, nr, u.Off(0, nr), work.Off(n), v, work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					Dlapmt(false, n, n, v, iwork)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x NR) or (M x N) or (M x M).
					if (nr < m) && !wntuf {
						Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
						if nr < n1 {
							Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
							Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
						}
					}
				}
			}

		} else {
			//            .. apply DGESVD to R [[this is the recommended option]]
			if wntvr || (nr == n) {
				//                .. copy R into [V] and overwrite V with the right singular vectors
				Dlacpy(Upper, nr, n, a, v)
				if nr > 1 {
					Dlaset(Lower, nr-1, nr-1, zero, zero, v.Off(1, 0))
				}
				//               .. the right singular vectors of R overwrite [V], the NR left
				//               singular vectors of R stored in [U](1:NR,1:NR)
				if info, err = Dgesvd('S', 'O', nr, n, v, s, u, v, work.Off(n), lwork-n); err != nil {
					panic(err)
				}
				Dlapmt(false, nr, n, v, iwork)
				//               .. now [V](1:NR,1:N) contains V(1:N,1:NR)**T
				//               .. assemble the left singular vector matrix U of dimensions
				//              (M x NR) or (M x N) or (M x M).
				if (nr < m) && !wntuf {
					Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
					if nr < n1 {
						Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
						Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
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
				if optratio*nr > n {
					Dlacpy(Upper, nr, n, a, v)
					if nr > 1 {
						Dlaset(Lower, nr-1, nr-1, zero, zero, v.Off(1, 0))
					}
					//              .. the right singular vectors of R overwrite [V], the NR left
					//                 singular vectors of R stored in [U](1:NR,1:NR)
					Dlaset(Full, n-nr, n, zero, zero, v.Off(nr, 0))
					if info, err = Dgesvd('S', 'O', n, n, v, s, u, v, work.Off(n), lwork-n); err != nil {
						panic(err)
					}
					Dlapmt(false, n, n, v, iwork)
					//                 .. now [V] contains the transposed matrix of the right
					//                 singular vectors of A. The leading N left singular vectors
					//                 are in [U](1:N,1:N)
					//                 .. assemble the left singular vector matrix U of dimensions
					//                 (M x N1), i.e. (M x N) or (M x M).
					if (n < m) && !wntuf {
						Dlaset(Full, m-n, n, zero, zero, u.Off(n, 0))
						if n < n1 {
							Dlaset(Full, n, n1-n, zero, zero, u.Off(0, n))
							Dlaset(Full, m-n, n1-n, zero, one, u.Off(n, n))
						}
					}
				} else {
					Dlacpy(Upper, nr, n, a, u.Off(nr, 0))
					if nr > 1 {
						Dlaset(Lower, nr-1, nr-1, zero, zero, u.Off(nr+2-1, 0))
					}
					if err = Dgelqf(nr, n, u.Off(nr, 0), work.Off(n), work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					Dlacpy(Lower, nr, nr, u.Off(nr, 0), v)
					if nr > 1 {
						Dlaset(Upper, nr-1, nr-1, zero, zero, v.Off(0, 1))
					}
					if info, err = Dgesvd('S', 'O', nr, nr, v, s, u, v, work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					Dlaset(Full, n-nr, nr, zero, zero, v.Off(nr, 0))
					Dlaset(Full, nr, n-nr, zero, zero, v.Off(0, nr))
					Dlaset(Full, n-nr, n-nr, zero, one, v.Off(nr, nr))
					if err = Dormlq(Right, NoTrans, n, n, nr, u.Off(nr, 0), work.Off(n), v, work.Off(n+nr), lwork-n-nr); err != nil {
						panic(err)
					}
					Dlapmt(false, n, n, v, iwork)
					//               .. assemble the left singular vector matrix U of dimensions
					//              (M x NR) or (M x N) or (M x M).
					if (nr < m) && !wntuf {
						Dlaset(Full, m-nr, nr, zero, zero, u.Off(nr, 0))
						if nr < n1 {
							Dlaset(Full, nr, n1-nr, zero, zero, u.Off(0, nr))
							Dlaset(Full, m-nr, n1-nr, zero, one, u.Off(nr, nr))
						}
					}
				}
			}
			//        .. end of the "R**T or R" branch
		}

		//           The Q matrix from the first QRF is built into the left singular
		//           vectors matrix U.
		if !wntuf {
			if err = Dormqr(Left, NoTrans, m, n1, n, a, work, u, work.Off(n), lwork-n); err != nil {
				panic(err)
			}
		}
		if rowprm && !wntuf {
			Dlaswp(n1, u, 1, m-1, *toSlice(iwork, n), -1)
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
	if nr < n {
		Dlaset(Full, n-nr, 1, zero, zero, s.Off(nr).Matrix(n, opts))
	}
	//     .. undo scaling; this may cause overflow in the largest singular
	//     values.
	if ascaled {
		if err = Dlascl('G', 0, 0, one, math.Sqrt(float64(m)), nr, 1, s.Matrix(n, opts)); err != nil {
			panic(err)
		}
	}
	if conda {
		rwork.Set(0, sconda)
	}
	rwork.Set(1, float64(p-nr))
	//     .. p-NR is the number of singular values that are computed as
	//     exact zeros in DGESVD() applied to the (possibly truncated)
	//     full row rank triangular (trapezoidal) factor of A.
	numrank = nr

	return
}
