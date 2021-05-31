package goblas

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/whipstein/golinalg/mat"

	"github.com/whipstein/golinalg/golapack/gltest"
)

func TestZblasLevel2(t *testing.T) {
	var fatal, null, reset, same, tran, upper bool
	var trans, transs mat.MatTrans
	var diag, diags mat.MatDiag
	var uplo, uplos mat.MatUplo
	var alpha, als, beta, bls, half, one, transl, zero complex128
	var err, errmax, ralpha, rals, rzero float64
	var i, ia, ib, ic, icd, ict, icu, ik, iku, im, in, incx, incxs, incy, incys, incmax, ix, iy, j, ja, jj, k, kl, kls, ks, ku, kus, laa, lda, ldas, lj, lx, ly, m, ml, ms, n, nalf, nargs, nbet, nc, nd, nidim, ninc, nk, nkb, nl, nmax, ns int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	half = (0.5 + 0.0*1i)
	rzero = 0.0
	nmax = 65
	incmax = 2

	ok := &gltest.Common.Infoc.Ok

	*ok = true
	isame := make([]bool, 13)
	idim := []int{0, 1, 2, 3, 5, 9}
	kb := []int{0, 1, 2, 4}
	inc := []int{1, 2, -1, -2}
	alf := cvdf([]complex128{0.0 + 0.0i, 1.0 + 0.0i, 0.7 - 0.9i})
	bet := cvdf([]complex128{0.0 + 0i, 1.0 + 0i, 1.3 - 1.1i})
	thresh := 16.0
	eps := epsilonf64()
	nidim = len(idim)
	nkb = len(kb)
	ninc = len(inc)
	nalf = len(alf.Data)
	nbet = len(bet.Data)
	snames := []string{"ZGEMV", "ZGBMV", "ZHRMV", "ZHBMV", "ZHPMV", "ZTRMV", "ZTBMV", "ZTPMV", "ZTRSV", "ZTBSV", "ZTPSV", "ZGERC", "ZGERU", "ZHER", "ZHPR", "ZHER2", "ZHPR2"}
	ichd := []mat.MatDiag{mat.Unit, mat.NonUnit}
	icht := []mat.MatTrans{mat.NoTrans, mat.Trans, mat.ConjTrans}
	ichu := []mat.MatUplo{mat.Upper, mat.Lower}
	aa := cvf(nmax * nmax)
	as := cvf(nmax * nmax)
	x := cvf(65)
	xs := cvf(nmax * incmax)
	xt := cvf(65)
	xx := cvf(nmax * incmax)
	y := cvf(65)
	ys := cvf(nmax * incmax)
	yt := cvf(65)
	yy := cvf(nmax * incmax)
	g := vf(65)
	a := cmf(65, 65, opts)

	n = minint(int(32), nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.SetRe(i-1, j-1, float64(maxint(i-j+1, 0)))
		}
		x.SetRe(j-1, float64(j))
		y.Set(j-1, zero)
	}
	for j = 1; j <= n; j++ {
		yy.SetRe(j-1, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	//     YY holds the exact result. On exit from ZMVCH YT holds
	//     the result computed by ZMVCH.
	trans = NoTrans
	zmvch(trans, n, n, one, a, nmax, x, 1, zero, y, 1, yt, g, yy, eps, &err, &fatal, true, t)
	same = lze(yy, yt, n)
	if !same || err != rzero {
		fmt.Printf(" ERROR IN ZMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
		panic("")
	}
	trans = Trans
	zmvch(trans, n, n, one, a, nmax, x, -1, zero, y, -1, yt, g, yy, eps, &err, &fatal, true, t)
	same = lze(yy, yt, n)
	if !same || err != rzero {
		fmt.Printf(" ERROR IN ZMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
		panic("")
	}

	for _, sname := range snames {
		fatal = false
		reset = true
		*ok = true
		errmax = 0.0

		if sname == "ZGEMV" || sname == "ZGBMV" {
			full := sname[2] == 'E'
			banded := sname[2] == 'B'
			for i := range isame {
				isame[i] = true
			}

			//     Define the number of arguments.
			if full {
				nargs = 11
			} else if banded {
				nargs = 13
			}
			//
			nc = 0
			reset = true
			errmax = rzero
			//
			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				nd = n/2 + 1
				//
				for im = 1; im <= 2; im++ {
					if im == 1 {
						m = maxint(n-nd, 0)
					}
					if im == 2 {
						m = minint(n+nd, nmax)
					}

					if banded {
						nk = nkb
					} else {
						nk = 1
					}
					for iku = 1; iku <= nk; iku++ {
						if banded {
							ku = kb[iku-1]
							kl = maxint(ku-1, 0)
						} else {
							ku = n - 1
							kl = m - 1
						}
						opts.Kl, opts.Ku = kl, ku
						//              Set LDA to 1 more than minimum value if room.
						if banded {
							lda = kl + ku + 1
						} else {
							lda = m
						}
						if lda < (nmax) {
							lda = lda + 1
						}
						//              Skip tests if not enough room.
						if lda > (nmax) {
							goto label1100
						}
						laa = lda * n
						null = n <= 0 || m <= 0

						//              Generate the matrix A.
						transl = zero
						zmakeL2([]byte(sname[1:3]), Full, NonUnit, m, n, a, nmax, aa, lda, kl, ku, &reset, transl)

						for ic = 1; ic <= 3; ic++ {
							trans = icht[ic-1]
							tran = trans == Trans || trans == ConjTrans

							if tran {
								ml = n
								nl = m
							} else {
								ml = m
								nl = n
							}

							for ix = 1; ix <= ninc; ix++ {
								incx = inc[ix-1]
								lx = absint(incx) * nl

								//                    Generate the vector X.
								transl = half
								zmakeL2([]byte("GE"), Full, NonUnit, 1, nl, x.CMatrix(1, opts), 1, xx, absint(incx), 0, nl-1, &reset, transl)
								if nl > 1 {
									x.Set(nl/2-1, zero)
									xx.Set(1+absint(incx)*(nl/2-1)-1, zero)
								}

								for iy = 1; iy <= ninc; iy++ {
									incy = inc[iy-1]
									ly = absint(incy) * ml

									for ia = 1; ia <= nalf; ia++ {
										alpha = alf.Get(ia - 1)

										for ib = 1; ib <= nbet; ib++ {
											beta = bet.Get(ib - 1)

											//                             Generate the vector Y.
											transl = zero
											zmakeL2([]byte("GE"), Full, NonUnit, 1, ml, y.CMatrix(1, opts), 1, yy, absint(incy), 0, ml-1, &reset, transl)

											nc = nc + 1

											//                             Save every datum before calling the
											//                             subroutine.
											transs = trans
											ms = m
											ns = n
											kls = kl
											kus = ku
											als = alpha
											as = aa.DeepCopy()
											ldas = lda
											xs = xx.DeepCopy()
											incxs = incx
											bls = beta
											ys = yy.DeepCopy()
											incys = incy

											//                             Call the subroutine.
											if full {
												Zgemv(trans, &m, &n, &alpha, aa.CMatrix(lda, opts), &lda, xx, &incx, &beta, yy, &incy)
											} else if banded {
												Zgbmv(trans, &m, &n, &kl, &ku, &alpha, aa.CMatrix(lda, opts), &lda, xx, &incx, &beta, yy, &incy)
											}

											//                             Check if error-exit was taken incorrectly.
											if !(*ok) {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
												(fatal) = true
												goto label1130
											}

											//                             See what data changed inside subroutines.
											isame[0] = trans == transs
											isame[1] = ms == m
											isame[2] = ns == n
											if full {
												isame[3] = als == alpha
												isame[4] = lze(as, aa, laa)
												isame[5] = ldas == lda
												isame[6] = lze(xs, xx, lx)
												isame[7] = incxs == incx
												isame[8] = bls == beta
												if null {
													isame[9] = lze(ys, yy, ly)
												} else {
													isame[9] = lzeres([]byte("GE"), Full, 1, ml, ys.CMatrix(absint(incy), opts), yy.CMatrix(absint(incy), opts), absint(incy))
												}
												isame[10] = incys == incy
											} else if banded {
												isame[3] = kls == kl
												isame[4] = kus == ku
												isame[5] = als == alpha
												isame[6] = lze(as, aa, laa)
												isame[7] = ldas == lda
												isame[8] = lze(xs, xx, lx)
												isame[9] = incxs == incx
												isame[10] = bls == beta
												if null {
													isame[11] = lze(ys, yy, ly)
												} else {
													isame[11] = lzeres([]byte("GE"), Full, 1, ml, ys.CMatrix(absint(incy), opts), yy.CMatrix(absint(incy), opts), absint(incy))
												}
												isame[12] = incys == incy
											}

											//                             If data was incorrectly changed, report
											//                             and return.
											same = true
											for i = 1; i <= nargs; i++ {
												same = same && isame[i-1]
												if !isame[i-1] {
													t.Fail()
													fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
												}
											}
											if !same {
												(fatal) = true
												goto label1130
											}

											if !null {
												//                                Check the result.
												zmvch(trans, m, n, alpha, a, nmax, x, incx, beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)
												errmax = maxf64(errmax, err)
												//                                If got really bad answer, report and
												//                                return.
												if fatal {
													goto label1130
												}
											} else {
												//                                Avoid repeating tests with M.le.0 or
												//                                N.le.0.
												goto label1110
											}

										}

									}

								}

							}

						}

					label1100:
					}

				label1110:
				}

			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label1130:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c',%3d,%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d)         .\n", nc, sname, trans.Byte(), m, n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" %6d: %6s('%c',%3d,%3d,%3d,%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d) .\n", nc, sname, trans.Byte(), m, n, kl, ku, alpha, lda, incx, beta, incy)
			}

		} else if sname == "ZHEMV" || sname == "ZHBMV" || sname == "ZHPMV" {
			full := sname[2] == 'E'
			banded := sname[2] == 'B'
			packed := sname[2] == 'P'
			//     Define the number of arguments.
			if full {
				nargs = 10
			} else if banded {
				nargs = 11
			} else if packed {
				nargs = 9
			}

			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]

				if banded {
					nk = nkb
				} else {
					nk = 1
				}
				for ik = 1; ik <= nk; ik++ {
					if banded {
						k = kb[ik-1]
					} else {
						k = n - 1
					}
					opts.Kl, opts.Ku = k, k
					//           Set LDA to 1 more than minimum value if room.
					if banded {
						lda = k + 1
					} else {
						lda = n
					}
					if lda < (nmax) {
						lda = lda + 1
					}
					//           Skip tests if not enough room.
					if lda > (nmax) {
						goto label2100
					}
					if packed {
						laa = (n * (n + 1)) / 2
					} else {
						laa = lda * n
					}
					null = n <= 0

					for ic = 1; ic <= 2; ic++ {
						uplo = ichu[ic-1]
						opts.Uplo = uplo

						//              Generate the matrix A.
						transl = zero
						zmakeL2([]byte(sname[1:3]), uplo, NonUnit, n, n, a, nmax, aa, lda, k, k, &reset, transl)

						for ix = 1; ix <= ninc; ix++ {
							incx = inc[ix-1]
							lx = absint(incx) * n

							//                 Generate the vector X.
							transl = half
							zmakeL2([]byte("GE"), Full, NonUnit, 1, n, x.CMatrix(1, opts), 1, xx, absint(incx), 0, n-1, &reset, transl)
							if n > 1 {
								x.Set(n/2-1, zero)
								xx.Set(1+absint(incx)*(n/2-1)-1, zero)
							}

							for iy = 1; iy <= ninc; iy++ {
								incy = inc[iy-1]
								ly = absint(incy) * n

								for ia = 1; ia <= nalf; ia++ {
									alpha = alf.Get(ia - 1)

									for ib = 1; ib <= nbet; ib++ {
										beta = bet.Get(ib - 1)

										//                          Generate the vector Y.
										transl = zero
										zmakeL2([]byte("GE"), Full, NonUnit, 1, n, y.CMatrix(1, opts), 1, yy, absint(incy), 0, n-1, &reset, transl)

										nc = nc + 1

										//                          Save every datum before calling the
										//                          subroutine.
										uplos = uplo
										ns = n
										ks = k
										als = alpha
										as = aa.DeepCopy()
										ldas = lda
										xs = xx.DeepCopy()
										incxs = incx
										bls = beta
										ys = yy.DeepCopy()
										incys = incy

										//                          Call the subroutine.
										if full {
											Zhemv(uplo, &n, &alpha, aa.CMatrix(lda, opts), &lda, xx, &incx, &beta, yy, &incy)
										} else if banded {
											Zhbmv(uplo, &n, &k, &alpha, aa.CMatrix(lda, opts), &lda, xx, &incx, &beta, yy, &incy)
										} else if packed {
											Zhpmv(uplo, &n, &alpha, aa, xx, &incx, &beta, yy, &incy)
										}

										//                          Check if error-exit was taken incorrectly.
										if !(*ok) {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
											(fatal) = true
											goto label2120
										}

										//                          See what data changed inside subroutines.
										isame[0] = uplo == uplos
										isame[1] = ns == n
										if full {
											isame[2] = als == alpha
											isame[3] = lze(as, aa, laa)
											isame[4] = ldas == lda
											isame[5] = lze(xs, xx, lx)
											isame[6] = incxs == incx
											isame[7] = bls == beta
											if null {
												isame[8] = lze(ys, yy, ly)
											} else {
												isame[8] = lzeres([]byte("GE"), Full, 1, n, ys.CMatrix(absint(incy), opts), yy.CMatrix(absint(incy), opts), absint(incy))
											}
											isame[9] = incys == incy
										} else if banded {
											isame[2] = ks == k
											isame[3] = als == alpha
											isame[4] = lze(as, aa, laa)
											isame[5] = ldas == lda
											isame[6] = lze(xs, xx, lx)
											isame[7] = incxs == incx
											isame[8] = bls == beta
											if null {
												isame[9] = lze(ys, yy, ly)
											} else {
												isame[9] = lzeres([]byte("GE"), Full, 1, n, ys.CMatrix(absint(incy), opts), yy.CMatrix(absint(incy), opts), absint(incy))
											}
											isame[10] = incys == incy
										} else if packed {
											isame[2] = als == alpha
											isame[3] = lze(as, aa, laa)
											isame[4] = lze(xs, xx, lx)
											isame[5] = incxs == incx
											isame[6] = bls == beta
											if null {
												isame[7] = lze(ys, yy, ly)
											} else {
												isame[7] = lzeres([]byte("GE"), Full, 1, n, ys.CMatrix(absint(incy), opts), yy.CMatrix(absint(incy), opts), absint(incy))
											}
											isame[8] = incys == incy
										}

										//                          If data was incorrectly changed, report and
										//                          return.
										same = true
										for i = 1; i <= nargs; i++ {
											same = same && isame[i-1]
											if !isame[i-1] {
												t.Fail()
												fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
											}
										}
										if !same {
											(fatal) = true
											goto label2120
										}

										if !null {
											//                             Check the result.
											zmvch(NoTrans, n, n, alpha, a, nmax, x, incx, beta, y, incy, yt, g, yy, eps, &err, &fatal, true, t)
											errmax = maxf64(errmax, err)
											//                             If got really bad answer, report and
											//                             return.
											if fatal {
												goto label2120
											}
										} else {
											//                             Avoid repeating tests with N.le.0
											goto label2110
										}

									}

								}

							}

						}

					}

				label2100:
				}

			label2110:
			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label2120:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d)             .\n", nc, sname, uplo.Byte(), n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" %6d: %6s('%c',%3d,%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d)         .\n", nc, sname, uplo.Byte(), n, k, alpha, lda, incx, beta, incy)
			} else if packed {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, AP, X,%2d,%4.1f, Y,%2d)                .\n", nc, sname, uplo.Byte(), n, alpha, incx, beta, incy)
			}

		} else if sname == "ZTRMV" || sname == "ZTBMV" || sname == "ZTPMV" || sname == "ZTRSV" || sname == "ZTBSV" || sname == "ZTPSV" {
			z := cvf(nmax)
			full := sname[2] == 'R'
			banded := sname[2] == 'B'
			packed := sname[2] == 'P'

			//     Define the number of arguments.
			opts.Style = mat.Triangular
			if full {
				nargs = 8
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 9
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 7
				opts.Storage = mat.Packed
			}

			nc = 0
			reset = true
			errmax = rzero
			//     Set up zero vector for ZMVCH.
			for i = 1; i <= (nmax); i++ {
				z.Set(i-1, zero)
			}

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]

				if banded {
					nk = nkb
				} else {
					nk = 1
				}
				for ik = 1; ik <= nk; ik++ {
					if banded {
						k = kb[ik-1]
					} else {
						k = n - 1
					}
					opts.Kl, opts.Ku = k, k
					//           Set LDA to 1 more than minimum value if room.
					if banded {
						lda = k + 1
					} else {
						lda = n
					}
					if lda < (nmax) {
						lda = lda + 1
					}
					//           Skip tests if not enough room.
					if lda > (nmax) {
						goto label3100
					}
					if packed {
						laa = (n * (n + 1)) / 2
					} else {
						laa = lda * n
					}
					null = n <= 0

					for icu = 1; icu <= 2; icu++ {
						uplo = ichu[icu-1]
						opts.Uplo = uplo

						for ict = 1; ict <= 3; ict++ {
							trans = icht[ict-1]

							for icd = 1; icd <= 2; icd++ {
								diag = ichd[icd-1]
								opts.Diag = diag

								//                    Generate the matrix A.
								transl = zero
								zmakeL2([]byte(sname[1:3]), uplo, diag, n, n, a, nmax, aa, lda, k, k, &reset, transl)

								for ix = 1; ix <= ninc; ix++ {
									incx = inc[ix-1]
									lx = absint(incx) * n

									//                       Generate the vector X.
									transl = half
									zmakeL2([]byte("GE"), Full, NonUnit, 1, n, x.CMatrix(1, opts), 1, xx, absint(incx), 0, n-1, &reset, transl)
									if n > 1 {
										x.Set(n/2-1, zero)
										xx.Set(1+absint(incx)*(n/2-1)-1, zero)
									}

									nc = nc + 1

									//                       Save every datum before calling the subroutine.
									uplos = uplo
									transs = trans
									diags = diag
									ns = n
									ks = k
									as = aa.DeepCopy()
									ldas = lda
									xs = xx.DeepCopy()
									incxs = incx

									//                       Call the subroutine.
									if string(sname[3:5]) == "MV" {
										if full {
											Ztrmv(uplo, trans, diag, &n, aa.CMatrix(lda, opts), &lda, xx, &incx)
										} else if banded {
											Ztbmv(uplo, trans, diag, &n, &k, aa.CMatrix(lda, opts), &lda, xx, &incx)
										} else if packed {
											Ztpmv(uplo, trans, diag, &n, aa, xx, &incx)
										}
									} else if string(sname[3:5]) == "SV" {
										if full {
											Ztrsv(uplo, trans, diag, &n, aa.CMatrix(lda, opts), &lda, xx, &incx)
										} else if banded {
											Ztbsv(uplo, trans, diag, &n, &k, aa.CMatrix(lda, opts), &lda, xx, &incx)
										} else if packed {
											Ztpsv(uplo, trans, diag, &n, aa, xx, &incx)
										}
									}

									//                       Check if error-exit was taken incorrectly.
									if !(*ok) {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										(fatal) = true
										goto label3120
									}

									//                       See what data changed inside subroutines.
									isame[0] = uplo == uplos
									isame[1] = trans == transs
									isame[2] = diag == diags
									isame[3] = ns == n
									if full {
										isame[4] = lze(as, aa, laa)
										isame[5] = ldas == lda
										if null {
											isame[6] = lze(xs, xx, lx)
										} else {
											isame[6] = lzeres([]byte("GE"), Full, 1, n, xs.CMatrix(absint(incx), opts), xx.CMatrix(absint(incx), opts), absint(incx))
										}
										isame[7] = incxs == incx
									} else if banded {
										isame[4] = ks == k
										isame[5] = lze(as, aa, laa)
										isame[6] = ldas == lda
										if null {
											isame[7] = lze(xs, xx, lx)
										} else {
											isame[7] = lzeres([]byte("GE"), Full, 1, n, xs.CMatrix(absint(incx), opts), xx.CMatrix(absint(incx), opts), absint(incx))
										}
										isame[8] = incxs == incx
									} else if packed {
										isame[4] = lze(as, aa, laa)
										if null {
											isame[5] = lze(xs, xx, lx)
										} else {
											isame[5] = lzeres([]byte("GE"), Full, 1, n, xs.CMatrix(absint(incx), opts), xx.CMatrix(absint(incx), opts), absint(incx))
										}
										isame[6] = incxs == incx
									}

									//                       If data was incorrectly changed, report and
									//                       return.
									same = true
									for i = 1; i <= nargs; i++ {
										same = same && isame[i-1]
										if !isame[i-1] {
											t.Fail()
											fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
										}
									}
									if !same {
										(fatal) = true
										goto label3120
									}

									if !null {
										if string(sname[3:5]) == "MV" {
											//                             Check the result.
											zmvch(trans, n, n, one, a, nmax, x, incx, zero, z, incx, xt, g, xx, eps, &err, &fatal, true, t)
										} else if string(sname[3:5]) == "SV" {
											//                             Compute approximation to original vector.
											for i = 1; i <= n; i++ {
												z.Set(i-1, xx.Get(1+(i-1)*absint(incx)-1))
												xx.Set(1+(i-1)*absint(incx)-1, x.Get(i-1))
											}
											zmvch(trans, n, n, one, a, nmax, z, incx, zero, x, incx, xt, g, xx, eps, &err, &fatal, false, t)
										}
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label3120
										}
									} else {
										//                          Avoid repeating tests with N.le.0.
										goto label3110
									}

								}

							}

						}

					}

				label3100:
				}

			label3110:
			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label3120:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c','%c','%c',%3d, A,%3d, X,%2d)                                   .\n", nc, sname, uplo.Byte(), trans.Byte(), diag.Byte(), n, lda, incx)
			} else if banded {
				fmt.Printf(" %6d: %6s('%c','%c','%c',%3d,%3d, A,%3d, X,%2d)                               .\n", nc, sname, uplo.Byte(), trans.Byte(), diag.Byte(), n, k, lda, incx)
			} else if packed {
				fmt.Printf(" %6d: %6s('%c','%c','%c',%3d, AP, X,%2d)                                      .\n", nc, sname, uplo.Byte(), trans.Byte(), diag.Byte(), n, incx)
			}

		} else if sname == "ZGERC" || sname == "ZGERU" {
			w := cvf(1)
			z := cvf(nmax)
			conj := sname[4] == 'C'
			//     Define the number of arguments.
			nargs = 9

			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				nd = n/2 + 1

				for im = 1; im <= 2; im++ {
					if im == 1 {
						m = maxint(n-nd, 0)
					}
					if im == 2 {
						m = minint(n+nd, nmax)
					}

					//           Set LDA to 1 more than minimum value if room.
					lda = m
					if lda < (nmax) {
						lda = lda + 1
					}
					//           Skip tests if not enough room.
					if lda > (nmax) {
						goto label4110
					}
					laa = lda * n
					null = n <= 0 || m <= 0
					//
					for ix = 1; ix <= ninc; ix++ {
						incx = inc[ix-1]
						lx = absint(incx) * m

						//              Generate the vector X.
						transl = half
						zmakeL2([]byte("GE"), Full, NonUnit, 1, m, x.CMatrix(1, opts), 1, xx, absint(incx), 0, m-1, &reset, transl)
						if m > 1 {
							x.Set(m/2-1, zero)
							xx.Set(1+absint(incx)*(m/2-1)-1, zero)
						}
						//
						for iy = 1; iy <= ninc; iy++ {
							incy = inc[iy-1]
							ly = absint(incy) * n

							//                 Generate the vector Y.
							transl = zero
							zmakeL2([]byte("GE"), Full, NonUnit, 1, n, y.CMatrix(1, opts), 1, yy, absint(incy), 0, n-1, &reset, transl)
							if n > 1 {
								y.Set(n/2-1, zero)
								yy.Set(1+absint(incy)*(n/2-1)-1, zero)
							}

							for ia = 1; ia <= nalf; ia++ {
								alpha = alf.Get(ia - 1)

								//                    Generate the matrix A.
								transl = zero
								zmakeL2([]byte(sname[1:3]), Full, NonUnit, m, n, a, nmax, aa, lda, m-1, n-1, &reset, transl)

								nc = nc + 1

								//                    Save every datum before calling the subroutine.
								ms = m
								ns = n
								als = alpha
								for i = 1; i <= laa; i++ {
									as.Set(i-1, aa.Get(i-1))
								}
								ldas = lda
								for i = 1; i <= lx; i++ {
									xs.Set(i-1, xx.Get(i-1))
								}
								incxs = incx
								for i = 1; i <= ly; i++ {
									ys.Set(i-1, yy.Get(i-1))
								}
								incys = incy

								//                    Call the subroutine.
								if conj {
									Zgerc(&m, &n, &alpha, xx, &incx, yy, &incy, aa.CMatrix(lda, opts), &lda)
								} else {
									Zgeru(&m, &n, &alpha, xx, &incx, yy, &incy, aa.CMatrix(lda, opts), &lda)
								}

								//                    Check if error-exit was taken incorrectly.
								if !(*ok) {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									(fatal) = true
									goto label4140
								}

								//                    See what data changed inside subroutine.
								isame[0] = ms == m
								isame[1] = ns == n
								isame[2] = als == alpha
								isame[3] = lze(xs, xx, lx)
								isame[4] = incxs == incx
								isame[5] = lze(ys, yy, ly)
								isame[6] = incys == incy
								if null {
									isame[7] = lze(as, aa, laa)
								} else {
									isame[7] = lzeres([]byte("GE"), Full, m, n, as.CMatrix(lda, opts), aa.CMatrix(lda, opts), lda)
								}
								isame[8] = ldas == lda

								//                    If data was incorrectly changed, report and return.
								same = true
								for i = 1; i <= nargs; i++ {
									same = same && isame[i-1]
									if !isame[i-1] {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
									}
								}
								if !same {
									(fatal) = true
									goto label4140
								}

								if !null {
									//                       Check the result column by column.
									if incx > 0 {
										for i = 1; i <= m; i++ {
											z.Set(i-1, x.Get(i-1))
										}
									} else {
										for i = 1; i <= m; i++ {
											z.Set(i-1, x.Get(m-i+1-1))
										}
									}
									for j = 1; j <= n; j++ {
										if incy > 0 {
											w.Set(0, y.Get(j-1))
										} else {
											w.Set(0, y.Get(n-j+1-1))
										}
										if conj {
											w.Set(0, w.GetConj(0))
										}
										zmvch(NoTrans, m, 1, alpha, z.CMatrix(nmax, opts), nmax, w, 1, one, a.CVector(0, j-1), 1, yt, g, aa.Off(1+(j-1)*lda-1), eps, &err, &fatal, true, t)
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label4130
										}
									}
								} else {
									//                       Avoid repeating tests with M.le.0 or N.le.0.
									goto label4110
								}

							}

						}

					}

				label4110:
				}

			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label4130:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			//
		label4140:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			fmt.Printf(" %6d: %6s(%3d,%3d,%4.1f, X,%2d, Y,%2d, A,%3d)                         .\n", nc, sname, m, n, alpha, incx, incy, lda)

		} else if sname == "ZHER" || sname == "ZHPR" {
			w := cvf(1)
			z := cvf(nmax)
			full := sname[2] == 'E'
			packed := sname[2] == 'P'
			//     Define the number of arguments.
			if full {
				nargs = 7
			} else if packed {
				nargs = 6
			}

			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				//        Set LDA to 1 more than minimum value if room.
				lda = n
				if lda < (nmax) {
					lda = lda + 1
				}
				//        Skip tests if not enough room.
				if lda > (nmax) {
					goto label5100
				}
				if packed {
					laa = (n * (n + 1)) / 2
				} else {
					laa = lda * n
				}

				for ic = 1; ic <= 2; ic++ {
					uplo = ichu[ic-1]
					upper = uplo == Upper

					for ix = 1; ix <= ninc; ix++ {
						incx = inc[ix-1]
						lx = absint(incx) * n

						//              Generate the vector X.
						transl = half
						zmakeL2([]byte("GE"), Full, NonUnit, 1, n, x.CMatrix(1, opts), 1, xx, absint(incx), 0, n-1, &reset, transl)
						if n > 1 {
							x.Set(n/2-1, zero)
							xx.Set(1+absint(incx)*(n/2-1)-1, zero)
						}
						//
						for ia = 1; ia <= nalf; ia++ {
							ralpha := real(alf.Get(ia - 1))
							alpha = complex(ralpha, rzero)
							null = n <= 0 || ralpha == rzero

							//                 Generate the matrix A.
							transl = zero
							zmakeL2([]byte(sname[1:3]), uplo, NonUnit, n, n, a, nmax, aa, lda, n-1, n-1, &reset, transl)

							nc = nc + 1

							//                 Save every datum before calling the subroutine.
							uplos = uplo
							ns = n
							rals = ralpha
							for i = 1; i <= laa; i++ {
								as.Set(i-1, aa.Get(i-1))
							}
							ldas = lda
							for i = 1; i <= lx; i++ {
								xs.Set(i-1, xx.Get(i-1))
							}
							incxs = incx

							//                 Call the subroutine.
							if full {
								Zher(uplo, &n, &ralpha, xx, &incx, aa.CMatrix(lda, opts), &lda)
							} else if packed {
								Zhpr(uplo, &n, &ralpha, xx, &incx, aa)
							}

							//                 Check if error-exit was taken incorrectly.
							if !(*ok) {
								t.Fail()
								fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
								(fatal) = true
								goto label5120
							}

							//                 See what data changed inside subroutines.
							isame[0] = uplo == uplos
							isame[1] = ns == n
							isame[2] = rals == ralpha
							isame[3] = lze(xs, xx, lx)
							isame[4] = incxs == incx
							if null {
								isame[5] = lze(as, aa, laa)
							} else {
								isame[5] = lzeres([]byte(sname[1:3]), uplo, n, n, as.CMatrix(lda, opts), aa.CMatrix(lda, opts), lda)
							}
							if !packed {
								isame[6] = ldas == lda
							}

							//                 If data was incorrectly changed, report and return.
							same = true
							for i = 1; i <= nargs; i++ {
								same = same && isame[i-1]
								if !isame[i-1] {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
								}
							}
							if !same {
								(fatal) = true
								goto label5120
							}

							if !null {
								//                    Check the result column by column.
								if incx > 0 {
									for i = 1; i <= n; i++ {
										z.Set(i-1, x.Get(i-1))
									}
								} else {
									for i = 1; i <= n; i++ {
										z.Set(i-1, x.Get(n-i+1-1))
									}
								}
								ja = 1
								for j = 1; j <= n; j++ {
									w.Set(0, z.GetConj(j-1))
									if upper {
										jj = 1
										lj = j
									} else {
										jj = j
										lj = n - j + 1
									}
									zmvch(NoTrans, lj, 1, alpha, z.CMatrixOff(jj-1, lj, opts), lj, w, 1, one, a.CVector(jj-1, j-1), 1, yt, g, aa.Off(ja-1), eps, &err, &fatal, true, t)
									if full {
										if upper {
											ja = ja + lda
										} else {
											ja = ja + lda + 1
										}
									} else {
										ja = ja + lj
									}
									errmax = maxf64(errmax, err)
									//                       If got really bad answer, report and return.
									if fatal {
										goto label5110
									}
								}
							} else {
								//                    Avoid repeating tests if N.le.0.
								if n <= 0 {
									goto label5100
								}
							}

						}

					}

				}

			label5100:
			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label5110:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)

		label5120:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, X,%2d, A,%3d)                                      .\n", nc, sname, uplo, n, ralpha, incx, lda)
			} else if packed {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, X,%2d, AP)                                         .\n", nc, sname, uplo, n, ralpha, incx)
			}

		} else if sname == "ZHER2" || sname == "ZHPR2" {
			w := cvf(2)
			z := cmf(nmax, 2, opts)
			full := sname[2] == 'E'
			packed := sname[2] == 'P'
			//     Define the number of arguments.
			if full {
				nargs = 9
			} else if packed {
				nargs = 8
			}

			nc = 0
			reset = true
			errmax = rzero

			for in = 1; in <= nidim; in++ {
				n = idim[in-1]
				//        Set LDA to 1 more than minimum value if room.
				lda = n
				if lda < (nmax) {
					lda = lda + 1
				}
				//        Skip tests if not enough room.
				if lda > (nmax) {
					goto label6140
				}
				if packed {
					laa = (n * (n + 1)) / 2
				} else {
					laa = lda * n
				}

				for ic = 1; ic <= 2; ic++ {
					uplo = ichu[ic-1]
					upper = uplo == Upper

					for ix = 1; ix <= ninc; ix++ {
						incx = inc[ix-1]
						lx = absint(incx) * n

						//              Generate the vector X.
						transl = half
						zmakeL2([]byte("GE"), Full, NonUnit, 1, n, x.CMatrix(1, opts), 1, xx, absint(incx), 0, n-1, &reset, transl)
						if n > 1 {
							x.Set(n/2-1, zero)
							xx.Set(1+absint(incx)*(n/2-1)-1, zero)
						}
						//
						for iy = 1; iy <= ninc; iy++ {
							incy = inc[iy-1]
							ly = absint(incy) * n

							//                 Generate the vector Y.
							transl = zero
							zmakeL2([]byte("GE"), Full, NonUnit, 1, n, y.CMatrix(1, opts), 1, yy, absint(incy), 0, n-1, &reset, transl)
							if n > 1 {
								y.Set(n/2-1, zero)
								yy.Set(1+absint(incy)*(n/2-1)-1, zero)
							}

							for ia = 1; ia <= nalf; ia++ {
								alpha = alf.Get(ia - 1)
								null = n <= 0 || alpha == zero

								//                    Generate the matrix A.
								transl = zero
								zmakeL2([]byte(sname[1:3]), uplo, NonUnit, n, n, a, nmax, aa, lda, n-1, n-1, &reset, transl)

								nc = nc + 1

								//                    Save every datum before calling the subroutine.
								uplos = uplo
								ns = n
								als = alpha
								for i = 1; i <= laa; i++ {
									as.Set(i-1, aa.Get(i-1))
								}
								ldas = lda
								for i = 1; i <= lx; i++ {
									xs.Set(i-1, xx.Get(i-1))
								}
								incxs = incx
								for i = 1; i <= ly; i++ {
									ys.Set(i-1, yy.Get(i-1))
								}
								incys = incy

								//                    Call the subroutine.
								if full {
									Zher2(uplo, &n, &alpha, xx, &incx, yy, &incy, aa.CMatrix(lda, opts), &lda)
								} else if packed {
									Zhpr2(uplo, &n, &alpha, xx, &incx, yy, &incy, aa)
								}

								//                    Check if error-exit was taken incorrectly.
								if !(*ok) {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									(fatal) = true
									goto label6160
								}

								//                    See what data changed inside subroutines.
								isame[0] = uplo == uplos
								isame[1] = ns == n
								isame[2] = als == alpha
								isame[3] = lze(xs, xx, lx)
								isame[4] = incxs == incx
								isame[5] = lze(ys, yy, ly)
								isame[6] = incys == incy
								if null {
									isame[7] = lze(as, aa, laa)
								} else {
									isame[7] = lzeres([]byte(sname[1:3]), uplo, n, n, as.CMatrix(lda, opts), aa.CMatrix(lda, opts), lda)
								}
								if !packed {
									isame[8] = ldas == lda
								}

								//                    If data was incorrectly changed, report and return.
								same = true
								for i = 1; i <= nargs; i++ {
									same = same && isame[i-1]
									if !isame[i-1] {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - PARAMETER NUMBER %2d WAS CHANGED INCORRECTLY *******\n", i)
									}
								}
								if !same {
									(fatal) = true
									goto label6160
								}

								if !null {
									//                       Check the result column by column.
									if incx > 0 {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(i-1))
										}
									} else {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(n-i+1-1))
										}
									}
									if incy > 0 {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 1, y.Get(i-1))
										}
									} else {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 1, y.Get(n-i+1-1))
										}
									}
									ja = 1
									for j = 1; j <= n; j++ {
										w.Set(0, alpha*z.GetConj(j-1, 1))
										w.Set(1, cmplx.Conj(alpha)*z.GetConj(j-1, 0))
										if upper {
											jj = 1
											lj = j
										} else {
											jj = j
											lj = n - j + 1
										}
										zmvch(NoTrans, lj, 2, one, z.Off(jj-1, 0), nmax, w, 1, one, a.CVector(jj-1, j-1), 1, yt, g, aa.Off(ja-1), eps, &err, &fatal, true, t)
										if full {
											if upper {
												ja = ja + lda
											} else {
												ja = ja + lda + 1
											}
										} else {
											ja = ja + lj
										}
										errmax = maxf64(errmax, err)
										//                          If got really bad answer, report and return.
										if fatal {
											goto label6150
										}
									}
								} else {
									//                       Avoid repeating tests with N.le.0.
									if n <= 0 {
										goto label6140
									}
								}

							}

						}

					}

				}

			label6140:
			}

			//     Report result.
			if errmax < thresh {
				fmt.Printf(" %6s PASSED THE COMPUTATIONAL TESTS (%6d CALLS)\n", sname, nc)
			} else {
				t.Fail()
				fmt.Printf(" %6s COMPLETED THE COMPUTATIONAL TESTS (%6d CALLS)\n ******* BUT WITH MAXIMUM TEST RATIO%8.2f - SUSPECT *******\n", sname, nc, errmax)
			}
			continue

		label6150:
			;
			fmt.Printf("      THESE ARE THE RESULTS FOR COLUMN %3d\n", j)
			//
		label6160:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, X,%2d, Y,%2d, A,%3d)                         .\n", nc, sname, uplo.Byte(), n, alpha, incx, incy, lda)
			} else if packed {
				fmt.Printf(" %6d: %6s('%c',%3d,%4.1f, X,%2d, Y,%2d, AP)                            .\n", nc, sname, uplo.Byte(), n, alpha, incx, incy)
			}

		}
	}
}

func zmakeL2(_type []byte, uplo mat.MatUplo, diag mat.MatDiag, m, n int, a *mat.CMatrix, nmax int, aa *mat.CVector, lda, kl, ku int, reset *bool, transl complex128) {
	var gen, lower, sym, tri, unit, upper bool
	var one, rogue, zero complex128
	var rrogue float64
	var i, i1, i2, i3, ibeg, iend, ioff, j, jj, kk int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rogue = (-1.0e10 + 1.0e10*1i)
	rrogue = -1.0e10

	gen = (_type)[0] == 'G'
	sym = (_type)[0] == 'H'
	tri = (_type)[0] == 'T'
	upper = (sym || tri) && uplo == Upper
	lower = (sym || tri) && uplo == Lower
	unit = tri && diag == Unit

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if gen || (upper && i <= j) || (lower && i >= j) {
				if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
					a.Set(i-1, j-1, zbeg(reset)+transl)
				} else {
					a.Set(i-1, j-1, zero)
				}
				if i != j {
					if sym {
						a.Set(j-1, i-1, a.GetConj(i-1, j-1))
					} else if tri {
						a.Set(j-1, i-1, zero)
					}
				}
			}
		}
		if sym {
			a.Set(j-1, j-1, a.GetReCmplx(j-1, j-1))
		}
		if tri {
			a.Set(j-1, j-1, a.Get(j-1, j-1)+one)
		}
		if unit {
			a.Set(j-1, j-1, one)
		}
	}

	//     Store elements in array AS in data structure required by routine.
	if string(_type) == "GE" {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i+(j-1)*lda-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
		}
	} else if string(_type) == "GB" {
		for j = 1; j <= n; j++ {
			for i1 = 1; i1 <= ku+1-j; i1++ {
				aa.Set(i1+(j-1)*lda-1, rogue)
			}
			for i2 = i1; i2 <= minint(kl+ku+1, ku+1+m-j); i2++ {
				aa.Set(i2+(j-1)*lda-1, a.Get(i2+j-ku-1-1, j-1))
			}
			for i3 = i2; i3 <= lda; i3++ {
				aa.Set(i3+(j-1)*lda-1, rogue)
			}
		}
	} else if string(_type) == "HE" || string(_type) == "TR" {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				if unit {
					iend = j - 1
				} else {
					iend = j
				}
			} else {
				if unit {
					ibeg = j + 1
				} else {
					ibeg = j
				}
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i+(j-1)*lda-1, a.Get(i-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			if sym {
				jj = j + (j-1)*lda
				aa.Set(jj-1, complex(real(aa.Get(jj-1)), rrogue))
			}
		}
	} else if string(_type) == "HB" || string(_type) == "TB" {
		for j = 1; j <= n; j++ {
			if upper {
				kk = kl + 1
				ibeg = maxint(1, kl+2-j)
				if unit {
					iend = kl
				} else {
					iend = kl + 1
				}
			} else {
				kk = 1
				if unit {
					ibeg = 2
				} else {
					ibeg = 1
				}
				iend = minint(kl+1, 1+m-j)
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i+(j-1)*lda-1, a.Get(i+j-kk-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i+(j-1)*lda-1, rogue)
			}
			if sym {
				jj = kk + (j-1)*lda
				aa.Set(jj-1, complex(real(aa.Get(jj-1)), rrogue))
			}
		}
	} else if string(_type) == "HP" || string(_type) == "TP" {
		ioff = 0
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = ibeg; i <= iend; i++ {
				ioff = ioff + 1
				aa.Set(ioff-1, a.Get(i-1, j-1))
				if i == j {
					if unit {
						aa.Set(ioff-1, rogue)
					}
					if sym {
						aa.Set(ioff-1, complex(real(aa.Get(ioff-1)), rrogue))
					}
				}
			}
		}
	}
}

func zmvch(trans mat.MatTrans, m, n int, alpha complex128, a *mat.CMatrix, nmax int, x *mat.CVector, incx int, beta complex128, y *mat.CVector, incy int, yt *mat.CVector, g *mat.Vector, yy *mat.CVector, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var ctran, tran bool
	var zero complex128
	var erri, rone, rzero float64
	var i, incxl, incyl, iy, j, jx, kx, ky, ml, nl int

	zero = (0.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	Abs1 := func(c complex128) float64 { return math.Abs(real(c)) + math.Abs(imag(c)) }

	tran = trans == Trans
	ctran = trans == ConjTrans
	if tran || ctran {
		ml = n
		nl = m
	} else {
		ml = m
		nl = n
	}
	if incx < 0 {
		kx = nl
		incxl = -1
	} else {
		kx = 1
		incxl = 1
	}
	if incy < 0 {
		ky = ml
		incyl = -1
	} else {
		ky = 1
		incyl = 1
	}

	//     Compute expected result in YT using data in A, X and Y.
	//     Compute gauges in G.
	iy = ky
	for i = 1; i <= ml; i++ {
		yt.Set(iy-1, zero)
		g.Set(iy-1, rzero)
		jx = kx
		if tran {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.Get(j-1, i-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+Abs1(a.Get(j-1, i-1))*Abs1(x.Get(jx-1)))
				jx = jx + incxl
			}
		} else if ctran {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.GetConj(j-1, i-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+Abs1(a.Get(j-1, i-1))*Abs1(x.Get(jx-1)))
				jx = jx + incxl
			}
		} else {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.Get(i-1, j-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+Abs1(a.Get(i-1, j-1))*Abs1(x.Get(jx-1)))
				jx = jx + incxl
			}
		}
		yt.Set(iy-1, alpha*yt.Get(iy-1)+beta*y.Get(iy-1))
		g.Set(iy-1, Abs1(alpha)*g.Get(iy-1)+Abs1(beta)*Abs1(y.Get(iy-1)))
		iy = iy + incyl
	}

	//     Compute the error ratio for this result.
	(*err) = rzero
	for i = 1; i <= ml; i++ {
		erri = cmplx.Abs(yt.Get(i-1)-yy.Get(1+(i-1)*absint(incy)-1)) / eps
		if g.Get(i-1) != rzero {
			erri = erri / g.Get(i-1)
		}
		(*err) = maxf64(*err, erri)
		if (*err)*math.Sqrt(eps) >= rone {
			goto label60
		}
	}
	//     If the loop completes, all results are at least half accurate.
	return

	//     Report fatal error.
label60:
	;
	t.Fail()
	(*fatal) = true
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n                       EXPECTED RESULT                    COMPUTED RESULT\n")
	for i = 1; i <= ml; i++ {
		if mv {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, yt.Get(i-1), yy.Get(1+(i-1)*absint(incy)-1))
		} else {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, yy.Get(1+(i-1)*absint(incy)-1), yt.Get(i-1))
		}
	}
}

func lze(ri, rj *mat.CVector, lr int) (lzeReturn bool) {
	var i int

	for i = 1; i <= lr; i++ {
		if ri.Get(i-1) != rj.Get(i-1) {
			goto label20
		}
	}
	lzeReturn = true
	goto label30
label20:
	;
	lzeReturn = false
label30:
	;
	return
}

func lzeres(_type []byte, uplo mat.MatUplo, m, n int, aa, as *mat.CMatrix, lda int) (lzeresReturn bool) {
	var upper bool
	var i, ibeg, iend, j int

	upper = uplo == Upper
	if string(_type) == "GE" {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					goto label70
				}
			}
		}
	} else if string(_type) == "HE" {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					goto label70
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					goto label70
				}
			}
		}
	}

	lzeresReturn = true
	goto label80
label70:
	;
	lzeresReturn = false
label80:
	;
	return
}

func zbeg(reset *bool) (zbegReturn complex128) {
	i := &common.begc.i
	ic := &common.begc.ic
	j := &common.begc.j
	mi := &common.begc.mi
	mj := &common.begc.mj

	if *reset {
		//        Initialize local variables.
		*mi = 891
		*mj = 457
		*i = 7
		*j = 7
		*ic = 0
		(*reset) = false
	}

	//     The sequence of values of I or J is bounded between 1 and 999.
	//     If initial I or J = 1,2,3,6,7 or 9, the period will be 50.
	//     If initial I or J = 4 or 8, the period will be 25.
	//     If initial I or J = 5, the period will be 10.
	//     IC is used to break up the period by skipping 1 value of I or J
	//     in 6.
	*ic++
label10:
	;
	*i *= *mi
	*j *= *mj
	*i -= 1000 * ((*i) / 1000)
	*j -= 1000 * ((*j) / 1000)
	if *ic >= 5 {
		*ic = 0
		goto label10
	}
	zbegReturn = complex(float64((*i)-500)/1001.0, float64((*j)-500)/1001.0)
	return
}
