package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dbdsvdx computes the singular value decomposition (SVD) of a real
//  N-by-N (upper or lower) bidiagonal matrix B, B = U * S * VT,
//  where S is a diagonal matrix with non-negative diagonal elements
//  (the singular values of B), and U and VT are orthogonal matrices
//  of left and right singular vectors, respectively.
//
//  Given an upper bidiagonal B with diagonal D = [ d_1 d_2 ... d_N ]
//  and superdiagonal E = [ e_1 e_2 ... e_N-1 ], DBDSVDX computes the
//  singular value decompositon of B through the eigenvalues and
//  eigenvectors of the N*2-by-N*2 tridiagonal matrix
//
//        |  0  d_1                |
//        | d_1  0  e_1            |
//  TGK = |     e_1  0  d_2        |
//        |         d_2  .   .     |
//        |              .   .   . |
//
//  If (s,u,v) is a singular triplet of B with ||u|| = ||v|| = 1, then
//  (+/-s,q), ||q|| = 1, are eigenpairs of TGK, with q = P * ( u' +/-v' ) /
//  math.Sqrt(2) = ( v_1 u_1 v_2 u_2 ... v_n u_n ) / math.Sqrt(2), and
//  P = [ e_{n+1} e_{1} e_{n+2} e_{2} ... ].
//
//  Given a TGK matrix, one can either a) compute -s,-v and change signs
//  so that the singular values (and corresponding vectors) are already in
//  descending order (as in DGESVD/DGESDD) or b) compute s,v and reorder
//  the values (and corresponding vectors). DBDSVDX implements a) by
//  calling DSTEVX (bisection plus inverse iteration, to be replaced
//  with a version of the Multiple Relative Robust Representation
//  algorithm. (See P. Willems and B. Lang, A framework for the MR^3
//  algorithm: theory and implementation, SIAM J. Sci. Comput.,
//  35:740-766, 2013.)
func Dbdsvdx(uplo mat.MatUplo, jobz, _range byte, n int, d, e *mat.Vector, vl, vu float64, il, iu, ns int, s *mat.Vector, z *mat.Matrix, work *mat.Vector, iwork *[]int) (info int, err error) {
	var allsv, indsv, lower, split, sveq0, valsv, wantz bool
	var rngvx byte
	var abstol, emin, eps, fudge, hndrd, meigth, mu, nrmu, nrmv, one, ortol, smax, smin, sqrt2, ten, thresh, tol, ulp, vltgk, vutgk, zero, zjtji float64
	var i, icolz, idbeg, idend, idptr, idtgk, ieptr, ietgk, iifail, iiwork, iltgk, irowu, irowv, irowz, isbeg, isplt, itemp, iutgk, j, k, nru, nrv, nsl, ntgk int

	zero = 0.0
	one = 1.0
	ten = 10.0
	hndrd = 100.0
	meigth = -0.1250
	fudge = 2.0

	//     Test the input parameters.
	allsv = _range == 'A'
	valsv = _range == 'V'
	indsv = _range == 'I'
	wantz = jobz == 'V'
	lower = uplo == Lower

	if uplo != Upper && !lower {
		err = fmt.Errorf("uplo != Upper && !lower: uplo=%s", uplo)
	} else if !(wantz || jobz == 'N') {
		err = fmt.Errorf("!(wantz || jobz == 'N'): jobz='%c'", jobz)
	} else if !(allsv || valsv || indsv) {
		err = fmt.Errorf("!(allsv || valsv || indsv): _range='%c'", _range)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if n > 0 {
		if valsv {
			if vl < zero {
				err = fmt.Errorf("vl < zero: _range='%c', vl=%v", _range, vl)
			} else if vu <= vl {
				err = fmt.Errorf("vu <= vl: _range='%c', vl=%v, vu=%v", _range, vl, vu)
			}
		} else if indsv {
			if il < 1 || il > max(1, n) {
				err = fmt.Errorf("il < 1 || il > max(1, n): _range='%c', n=%v, il=%v", _range, n, il)
			} else if iu < min(n, il) || iu > n {
				err = fmt.Errorf("iu < min(n, il) || iu > n: _range='%c', n=%v, il=%v, iu=%v", _range, n, il, iu)
			}
		}
	}
	if err == nil {
		if z.Rows < 1 || (wantz && z.Rows < n*2) {
			err = fmt.Errorf("z.Rows < 1 || (wantz && z.Rows < n*2): jobz='%c', z.Rows=%v, n=%v", jobz, z.Rows, n)
		}
	}

	if err != nil {
		gltest.Xerbla2("Dbdsvdx", err)
		return
	}

	//     Quick return if possible (N.LE.1)
	ns = 0
	if n == 0 {
		return
	}

	if n == 1 {
		if allsv || indsv {
			ns = 1
			s.Set(0, math.Abs(d.Get(0)))
		} else {
			if vl < math.Abs(d.Get(0)) && vu >= math.Abs(d.Get(0)) {
				ns = 1
				s.Set(0, math.Abs(d.Get(0)))
			}
		}
		if wantz {
			z.Set(0, 0, math.Copysign(one, d.Get(0)))
			z.Set(1, 0, one)
		}
		return
	}

	abstol = 2 * Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)
	eps = Dlamch(Epsilon)
	sqrt2 = math.Sqrt(2)
	ortol = math.Sqrt(ulp)

	//     Criterion for splitting is taken from DBDSQR when singular
	//     values are computed to relative accuracy TOL. (See J. Demmel and
	//     W. Kahan, Accurate singular values of bidiagonal matrices, SIAM
	//     J. Sci. and Stat. Comput., 11:873–912, 1990.)
	tol = math.Max(ten, math.Min(hndrd, math.Pow(eps, meigth))) * eps

	//     Compute approximate maximum, minimum singular values.
	i = d.Iamax(n, 1)
	smax = math.Abs(d.Get(i - 1))
	i = e.Iamax(n-1, 1)
	smax = math.Max(smax, math.Abs(e.Get(i-1)))

	//     Compute threshold for neglecting D's and E's.
	smin = math.Abs(d.Get(0))
	if smin != zero {
		mu = smin
		for i = 2; i <= n; i++ {
			mu = math.Abs(d.Get(i-1)) * (mu / (mu + math.Abs(e.Get(i-1-1))))
			smin = math.Min(smin, mu)
			if smin == zero {
				break
			}
		}
	}
	smin = smin / math.Sqrt(float64(n))
	thresh = tol * smin

	//     Check for zeros in D and E (splits), i.e. submatrices.
	for i = 1; i <= n-1; i++ {
		if math.Abs(d.Get(i-1)) <= thresh {
			d.Set(i-1, zero)
		}
		if math.Abs(e.Get(i-1)) <= thresh {
			e.Set(i-1, zero)
		}
	}
	if math.Abs(d.Get(n-1)) <= thresh {
		d.Set(n-1, zero)
	}

	//     Pointers for arrays used by DSTEVX.
	idtgk = 1
	ietgk = idtgk + n*2
	itemp = ietgk + n*2
	iifail = 1
	iiwork = iifail + n*2

	//     Set RNGVX, which corresponds to RANGE for DSTEVX in TGK mode.
	//     VL,VU or IL,IU are redefined to conform to implementation a)
	//     described in the leading comments.
	iltgk = 0
	iutgk = 0
	vltgk = zero
	vutgk = zero

	if allsv {
		//        All singular values will be found. We aim at -s (see
		//        leading comments) with RNGVX = 'I'. IL and IU are set
		//        later (as ILTGK and IUTGK) according to the dimension
		//        of the active submatrix.
		rngvx = 'I'
		if wantz {
			Dlaset(Full, n*2, n+1, zero, zero, z)
		}
	} else if valsv {
		//        Find singular values in a half-open interval. We aim
		//        at -s (see leading comments) and we swap VL and VU
		//        (as VUTGK and VLTGK), changing their signs.
		rngvx = 'V'
		vltgk = -vu
		vutgk = -vl
		for _i := idtgk; _i <= idtgk+2*n-1; _i++ {
			work.Set(_i-1, zero)
		}
		work.Off(ietgk-1).Copy(n, d, 1, 2)
		work.Off(ietgk).Copy(n-1, e, 1, 2)
		if ns, info, err = Dstevx('N', 'V', n*2, work.Off(idtgk-1), work.Off(ietgk-1), vltgk, vutgk, iltgk, iltgk, abstol, s, z, work.Off(itemp-1), toSlice(iwork, iiwork-1), toSlice(iwork, iifail-1)); err != nil {
			panic(err)
		}
		if ns == 0 {
			return
		} else {
			if wantz {
				Dlaset(Full, n*2, ns, zero, zero, z)
			}
		}
	} else if indsv {
		//        Find the IL-th through the IU-th singular values. We aim
		//        at -s (see leading comments) and indices are mapped into
		//        values, therefore mimicking DSTEBZ, where
		//
		//        GL = GL - FUDGE*TNORM*ULP*N - FUDGE*TWO*PIVMIN
		//        GU = GU + FUDGE*TNORM*ULP*N + FUDGE*PIVMIN
		iltgk = il
		iutgk = iu
		rngvx = 'V'
		for _i := idtgk; _i <= idtgk+2*n-1; _i++ {
			work.Set(_i-1, zero)
		}
		work.Off(ietgk-1).Copy(n, d, 1, 2)
		work.Off(ietgk).Copy(n-1, e, 1, 2)
		if ns, info, err = Dstevx('N', 'I', n*2, work.Off(idtgk-1), work.Off(ietgk-1), vltgk, vltgk, iltgk, iltgk, abstol, s, z, work.Off(itemp-1), toSlice(iwork, iiwork-1), toSlice(iwork, iifail-1)); err != nil {
			panic(err)
		}
		vltgk = s.Get(0) - fudge*smax*ulp*float64(n)
		for _i := idtgk; _i <= idtgk+2*n-1; _i++ {
			work.Set(_i-1, zero)
		}
		work.Off(ietgk-1).Copy(n, d, 1, 2)
		work.Off(ietgk).Copy(n-1, e, 1, 2)
		if ns, info, err = Dstevx('N', 'I', n*2, work.Off(idtgk-1), work.Off(ietgk-1), vutgk, vutgk, iutgk, iutgk, abstol, s, z, work.Off(itemp-1), toSlice(iwork, iiwork-1), toSlice(iwork, iifail-1)); err != nil {
			panic(err)
		}
		vutgk = s.Get(0) + fudge*smax*ulp*float64(n)
		vutgk = math.Min(vutgk, zero)

		//        If VLTGK=VUTGK, DSTEVX returns an error message,
		//        so if needed we change VUTGK slightly.
		if vltgk == vutgk {
			vltgk = vltgk - tol
		}

		if wantz {
			Dlaset(Full, n*2, iu-il+1, zero, zero, z)
		}
	}

	//     Initialize variables and pointers for S, Z, and WORK.
	//
	//     NRU, NRV: number of rows in U and V for the active submatrix
	//     IDBEG, ISBEG: offsets for the entries of D and S
	//     IROWZ, ICOLZ: offsets for the rows and columns of Z
	//     IROWU, IROWV: offsets for the rows of U and V
	ns = 0
	nru = 0
	nrv = 0
	idbeg = 1
	isbeg = 1
	irowz = 1
	icolz = 1
	irowu = 2
	irowv = 1
	split = false
	sveq0 = false

	//     Form the tridiagonal TGK matrix.
	s.Set(0, zero)
	work.Set(ietgk+2*n-1-1, zero)
	for _i := idtgk; _i <= idtgk+2*n-1; _i++ {
		work.Set(_i-1, zero)
	}
	work.Off(ietgk-1).Copy(n, d, 1, 2)
	work.Off(ietgk).Copy(n-1, e, 1, 2)

	//     Check for splits in two levels, outer level
	//     in E and inner level in D.
	for ieptr = 2; ieptr <= n*2; ieptr += 2 {
		if work.Get(ietgk+ieptr-1-1) == zero {
			//           Split in E (this piece of B is square) or bottom
			//           of the (input bidiagonal) matrix.
			isplt = idbeg
			idend = ieptr - 1
			for idptr = idbeg; idptr <= idend; idptr += 2 {
				if work.Get(ietgk+idptr-1-1) == zero {
					//                 Split in D (rectangular submatrix). Set the number
					//                 of rows in U and V (NRU and NRV) accordingly.
					if idptr == idbeg {
						//                    D=0 at the top.
						sveq0 = true
						if idbeg == idend {
							nru = 1
							nrv = 1
						}
					} else if idptr == idend {
						//                    D=0 at the bottom.
						sveq0 = true
						nru = (idend-isplt)/2 + 1
						nrv = nru
						if isplt != idbeg {
							nru = nru + 1
						}
					} else {
						if isplt == idbeg {
							//                       Split: top rectangular submatrix.
							nru = (idptr - idbeg) / 2
							nrv = nru + 1
						} else {
							//                       Split: middle square submatrix.
							nru = (idptr-isplt)/2 + 1
							nrv = nru
						}
					}
				} else if idptr == idend {
					//                 Last entry of D in the active submatrix.
					if isplt == idbeg {
						//                    No split (trivial case).
						nru = (idend-idbeg)/2 + 1
						nrv = nru
					} else {
						//                    Split: bottom rectangular submatrix.
						nrv = (idend-isplt)/2 + 1
						nru = nrv + 1
					}
				}

				ntgk = nru + nrv

				if ntgk > 0 {
					//                 Compute eigenvalues/vectors of the active
					//                 submatrix according to RANGE:
					//                 if RANGE='A' (ALLSV) then RNGVX = 'I'
					//                 if RANGE='V' (VALSV) then RNGVX = 'V'
					//                 if RANGE='I' (INDSV) then RNGVX = 'V'
					iltgk = 1
					iutgk = ntgk / 2
					if allsv || vutgk == zero {
						if sveq0 || smin < eps || (ntgk%2) > 0 {
							//                        Special case: eigenvalue equal to zero or very
							//                        small, additional eigenvector is needed.
							iutgk = iutgk + 1
						}
					}

					//                 Workspace needed by DSTEVX:
					//                 WORK( ITEMP: ): 2*5*NTGK
					//                 IWORK( 1: ): 2*6*NTGK
					if nsl, info, err = Dstevx(jobz, rngvx, ntgk, work.Off(idtgk+isplt-1-1), work.Off(ietgk+isplt-1-1), vltgk, vutgk, iltgk, iutgk, abstol, s.Off(isbeg-1), z.Off(irowz-1, icolz-1), work.Off(itemp-1), toSlice(iwork, iiwork-1), toSlice(iwork, iifail-1)); err != nil {
						panic(err)
					}
					if info != 0 {
						//                    Exit with the error code from DSTEVX.
						return
					}
					emin = 0
					if isbeg < isbeg+nsl-1 {
						_x := s.Get(isbeg)
						for _i := isbeg; _i < isbeg+nsl-1; _i++ {
							_x = math.Max(_x, s.Get(_i))
						}
						emin = math.Abs(_x)
					}

					if nsl > 0 && wantz {
						//                    Normalize u=Z([2,4,...],:) and v=Z([1,3,...],:),
						//                    changing the sign of v as discussed in the leading
						//                    comments. The norms of u and v may be (slightly)
						//                    different from 1/math.Sqrt(2) if the corresponding
						//                    eigenvalues are very small or too close. We check
						//                    those norms and, if needed, reorthogonalize the
						//                    vectors.
						if nsl > 1 && vutgk == zero && (ntgk%2) == 0 && emin == 0 && !split {
							//                       D=0 at the top or bottom of the active submatrix:
							//                       one eigenvalue is equal to zero; concatenate the
							//                       eigenvectors corresponding to the two smallest
							//                       eigenvalues.
							for _i := irowz; _i <= irowz+ntgk-1; _i++ {
								z.Set(_i-1, icolz+nsl-2-1, z.Get(_i-1, icolz+nsl-2-1)+z.Get(_i-1, icolz+nsl-1-1))
								z.Set(_i-1, icolz+nsl-1-1, zero)
							}
							//                       IF( IUTGK*2.GT.NTGK ) THEN
							//                          Eigenvalue equal to zero or very small.
							//                          NSL = NSL - 1
							//                       END IF
						}

						for i = 0; i <= min(nsl-1, nru-1); i++ {
							nrmu = z.Off(irowu-1, icolz+i-1).Vector().Nrm2(nru, 2)
							if nrmu == zero {
								info = n*2 + 1
								return
							}
							z.Off(irowu-1, icolz+i-1).Vector().Scal(nru, one/nrmu, 2)
							if nrmu != one && math.Abs(nrmu-ortol)*sqrt2 > one {
								for j = 0; j <= i-1; j++ {
									zjtji = -z.Off(irowu-1, icolz+i-1).Vector().Dot(nru, z.Off(irowu-1, icolz+j-1).Vector(), 2, 2)
									z.Off(irowu-1, icolz+i-1).Vector().Axpy(nru, zjtji, z.Off(irowu-1, icolz+j-1).Vector(), 2, 2)
								}
								nrmu = z.Off(irowu-1, icolz+i-1).Vector().Nrm2(nru, 2)
								z.Off(irowu-1, icolz+i-1).Vector().Scal(nru, one/nrmu, 2)
							}
						}
						for i = 0; i <= min(nsl-1, nrv-1); i++ {
							nrmv = z.Off(irowv-1, icolz+i-1).Vector().Nrm2(nrv, 2)
							if nrmv == zero {
								info = n*2 + 1
								return
							}
							z.Off(irowv-1, icolz+i-1).Vector().Scal(nrv, -one/nrmv, 2)
							if nrmv != one && math.Abs(nrmv-ortol)*sqrt2 > one {
								for j = 0; j <= i-1; j++ {
									zjtji = -z.Off(irowv-1, icolz+i-1).Vector().Dot(nrv, z.Off(irowv-1, icolz+j-1).Vector(), 2, 2)
									z.Off(irowv-1, icolz+i-1).Vector().Axpy(nru, zjtji, z.Off(irowv-1, icolz+j-1).Vector(), 2, 2)
								}
								nrmv = z.Off(irowv-1, icolz+i-1).Vector().Nrm2(nrv, 2)
								z.Off(irowv-1, icolz+i-1).Vector().Scal(nrv, one/nrmv, 2)
							}
						}
						if vutgk == zero && idptr < idend && (ntgk%2) > 0 {
							//                       D=0 in the middle of the active submatrix (one
							//                       eigenvalue is equal to zero): save the corresponding
							//                       eigenvector for later use (when bottom of the
							//                       active submatrix is reached).
							split = true
							for _i := irowz; _i <= irowz+ntgk-1; _i++ {
								z.Set(_i-1, n, z.Get(_i-1, ns+nsl-1))
								z.Set(_i-1, ns+nsl-1, zero)
							}
						}
					}
					//!** WANTZ **!

					nsl = min(nsl, nru)
					sveq0 = false

					//                 Absolute values of the eigenvalues of TGK.
					for i = 0; i <= nsl-1; i++ {
						s.Set(isbeg+i-1, math.Abs(s.Get(isbeg+i-1)))
					}

					//                 Update pointers for TGK, S and Z.
					isbeg = isbeg + nsl
					irowz = irowz + ntgk
					icolz = icolz + nsl
					irowu = irowz
					irowv = irowz + 1
					isplt = idptr + 1
					ns = ns + nsl
					nru = 0
					nrv = 0
				}
				//!** NTGK.GT.0 **!

				if irowz < n*2 && wantz {
					for _i := 1; _i <= irowz-1; _i++ {
						z.Set(_i-1, icolz-1, zero)
					}
				}
			}
			//!** IDPTR loop **!

			if split && wantz {
				//              Bring back eigenvector corresponding
				//              to eigenvalue equal to zero.
				for _i := idbeg; _i <= idend-ntgk+1; _i++ {
					z.Set(_i-1, isbeg-1-1, z.Get(_i-1, isbeg-1-1)+z.Get(_i-1, n))
					z.Set(_i-1, n, 0)
				}
			}
			irowv = irowv - 1
			irowu = irowu + 1
			idbeg = ieptr + 1
			sveq0 = false
			split = false
		}
		//!** Check for split in E **!

	}
	//!** IEPTR loop **!

	//     Sort the singular values into decreasing order (insertion sort on
	//     singular values, but only one transposition per singular vector)
	for i = 1; i <= ns-1; i++ {
		k = 1
		smin = s.Get(0)
		for j = 2; j <= ns+1-i; j++ {
			if s.Get(j-1) <= smin {
				k = j
				smin = s.Get(j - 1)
			}
		}
		if k != ns+1-i {
			s.Set(k-1, s.Get(ns+1-i-1))
			s.Set(ns+1-i-1, smin)
			if wantz {
				z.Off(0, ns+1-i-1).Vector().Swap(n*2, z.Off(0, k-1).Vector(), 1, 1)
			}
		}
	}

	//     If RANGE=I, check for singular values/vectors to be discarded.
	if indsv {
		k = iu - il + 1
		if k < ns {
			for _i := k + 1; _i <= ns; _i++ {
				s.Set(_i-1, zero)
				if wantz {
					for _j := 1; _j <= n*2; _j++ {
						z.Set(_j-1, _i-1, zero)
					}
				}
			}
			ns = k
		}
	}

	//     Reorder Z: U = Z( 1:N,1:NS ), V = Z( N+1:N*2,1:NS ).
	//     If B is a lower diagonal, swap U and V.
	if wantz {
		for i = 1; i <= ns; i++ {
			work.Copy(n*2, z.Off(0, i-1).Vector(), 1, 1)
			if lower {
				z.Off(n, i-1).Vector().Copy(n, work.Off(1), 2, 1)
				z.Off(0, i-1).Vector().Copy(n, work, 2, 1)
			} else {
				z.Off(0, i-1).Vector().Copy(n, work.Off(1), 2, 1)
				z.Off(n, i-1).Vector().Copy(n, work, 2, 1)
			}
		}
	}

	return
}
