package goblas

import (
	"fmt"
	"math"
	"math/cmplx"
	"reflect"
	"testing"

	"github.com/whipstein/golinalg/mat"
)

func TestZblasLevel2(t *testing.T) {
	var fatal, null, reset, same, tran, upper bool
	var trans, transs mat.MatTrans
	var diag, diags mat.MatDiag
	var uplo, uplos mat.MatUplo
	var alpha, als, beta, bls, one, zero complex128
	var err, errmax, ralpha, rals, rzero float64
	var i, ia, ib, ic, icd, ict, icu, ik, iku, im, in, incx, incxs, incy, incys, incmax, ix, iy, j, ja, jj, k, kl, kls, ks, ku, kus, lda, ldas, lj, lx, ly, m, ml, ms, n, nalf, nargs, nbet, nc, nd, nidim, ninc, nk, nkb, nl, nmax, ns int
	_ = iku
	_ = kl
	_ = kls
	_ = ku
	_ = kus
	_ = ml
	_ = nl
	var err2 error
	_ = err2
	ok := true

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rzero = 0.0
	nmax = 65
	incmax = 2

	isame := make([]bool, 13)
	idim := []int{0, 1, 2, 3, 5, 9}
	kb := []int{0, 1, 2, 4}
	inc := []int{1, 2, -1, -2}
	alf := cvdf([]complex128{0.0 + 0.0i, 1.0 + 0.0i, 0.7 - 0.9i})
	bet := cvdf([]complex128{0.0 + 0i, 1.0 + 0i, 1.3 - 1.1i})
	_alf := []complex128{0.0 + 0.0i, 1.0 + 0.0i, 0.7 - 0.9i}
	_bet := []complex128{0.0 + 0i, 1.0 + 0i, 1.3 - 1.1i}
	thresh := 16.0
	eps := epsilonf64()
	nidim = len(idim)
	nkb = len(kb)
	ninc = len(inc)
	nalf = len(alf.Data)
	nbet = len(bet.Data)
	snames := []string{"Zgemv", "Zgbmv", "Zhemv", "Zhbmv", "Zhpmv", "Ztrmv", "Ztbmv", "Ztpmv", "Ztrsv", "Ztbsv", "Ztpsv", "Zgerc", "Zgeru", "Zher", "Zhpr", "Zher2", "Zhpr2"}
	ichd := []mat.MatDiag{mat.Unit, mat.NonUnit}
	icht := []mat.MatTrans{mat.NoTrans, mat.Trans, mat.ConjTrans}
	ichu := []mat.MatUplo{mat.Upper, mat.Lower}
	optsfull := opts.DeepCopy()
	optsge := opts.DeepCopy()
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
	aa := cmf(65, 65, opts)
	as := cmf(65, 65, opts)
	fmt.Printf("\n***** ZBLAS Level 2 Tests *****\n")

	n = min(int(32), nmax)
	for j = 1; j <= n; j++ {
		for i = 1; i <= n; i++ {
			a.SetRe(i-1, j-1, float64(max(i-j+1, 0)))
		}
		x.SetRe(j-1, float64(j))
		y.Set(j-1, zero)
	}
	for j = 1; j <= n; j++ {
		yy.SetRe(j-1, float64(j*((j+1)*j))/2-float64((j+1)*j*(j-1))/3)
	}
	//     YY holds the exact result. On exit from ZMVCH YT holds
	//     the result computed by ZMVCH.
	zmvch(NoTrans, n, n, one, a, nmax, x, zero, y, yt, g, yy, eps, &err, &fatal, true, t)
	if err != rzero {
		fmt.Printf(" ERROR IN ZMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
		panic("")
	}
	x.Inc, y.Inc = -1, -1
	zmvch(Trans, n, n, one, a, nmax, x, zero, y, yt, g, yy, eps, &err, &fatal, true, t)
	if err != rzero {
		fmt.Printf(" ERROR IN ZMVCH -  IN-LINE DOT PRODUCTS ARE BEING EVALUATED WRONGLY.\n ZMVCH WAS CALLED WITH TRANS = %c AND RETURNED SAME =  %t AND ERR = %12.3f.\n THIS MAY BE DUE TO FAULTS IN THE ARITHMETIC OR THE COMPILER.\n ******* TESTS ABANDONED *******\n", trans, same, err)
		panic("")
	}

	for _, sname := range snames {
		fatal = false
		reset = true
		ok = true
		errmax = 0.0

		if sname == "Zgemv" || sname == "Zgbmv" {
			var i, iku, im, incx, incy, kl, kls, ku, kus, lda, m, ml, ms, n, nargs, nc, nd, nk, nl, ns int
			var full bool = sname[2] == 'e'
			var banded bool = sname[2] == 'b'

			for i := range isame {
				isame[i] = true
			}

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			opts.Style = mat.General
			optsfull.Style = mat.General
			if full {
				nargs = 11
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 13
				opts.Storage = mat.Banded
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				opts.Major = maj
				optsfull.Major = maj

				for _, n = range idim {
					nd = n/2 + 1

					for im = 1; im <= 2; im++ {
						if im == 1 {
							m = max(n-nd, 0)
						}
						if im == 2 {
							m = min(n+nd, nmax)
						}

						if banded {
							nk = nkb
						} else {
							nk = 1
						}

						for iku = 1; iku <= nk; iku++ {
							if banded {
								ku = kb[iku-1]
								kl = max(ku-1, 0)
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
							if lda < nmax {
								lda++
							}
							//              Skip tests if not enough room.
							if lda > nmax {
								continue
							}
							null = n <= 0 || m <= 0

							//              Generate the matrix A.
							a = cmf(nmax, n, optsfull)
							aa = cmf(lda, n, opts)
							zmakeL2M(m, n, a, nmax, aa, lda, kl, ku, &reset, 0.0)

							for _, trans := range mat.IterMatTrans() {
								tran = trans.IsTrans()

								if tran {
									ml = n
									nl = m
								} else {
									ml = m
									nl = n
								}

								for _, incx = range inc {
									lx := abs(incx) * nl

									//                    Generate the vector X.
									x = cvf(nl, incx)
									xx = cvf(lx, incx)
									zmakeL2V(1, nl, x, 1, xx, abs(incx), 0, nl-1, &reset, 0.5)
									if nl > 1 {
										x.Set(nl/2-1, 0)
										xx.Set(abs(incx)*(nl/2-1), 0)
									}

									for _, incy = range inc {
										ly := abs(incy) * ml

										for _, alpha = range _alf {

											for _, beta = range _bet {
												//                             Generate the vector Y.
												y = cvf(ml, incy)
												yy = cvf(ly, incy)
												zmakeL2V(1, ml, y, 1, yy, abs(incy), 0, ml-1, &reset, 0.0)

												nc++

												//                             Save every datum before calling the
												//                             subroutine.
												transs := trans
												ms = m
												ns = n
												kls = kl
												kus = ku
												als = alpha
												as = aa.DeepCopy()
												xs := xx.DeepCopy()
												bls = beta
												ys := yy.DeepCopy()

												//                             Call the subroutine.
												if full {
													err2 = Zgemv(trans, m, n, alpha, aa, xx, beta, yy)
												} else if banded {
													err2 = Zgbmv(trans, m, n, kl, ku, alpha, aa, xx, beta, yy)
												}

												//                             Check if error-exit was taken incorrectly.
												if !ok {
													t.Fail()
													fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
													fatal = true
													goto label130
												}

												//                             See what data changed inside subroutines.
												isame[0] = trans == transs
												isame[1] = ms == m
												isame[2] = ns == n
												if full {
													isame[3] = als == alpha
													isame[4] = reflect.DeepEqual(*aa, *as)
													isame[6] = reflect.DeepEqual(*xx, *xs)
													isame[8] = bls == beta
													if null {
														isame[9] = reflect.DeepEqual(*yy, *ys)
													} else {
														isame[9] = lzeresV(mat.General, mat.Full, 1, ml, ys, yy, abs(incy))
													}
												} else if banded {
													isame[3] = kls == kl
													isame[4] = kus == ku
													isame[5] = als == alpha
													isame[6] = reflect.DeepEqual(*aa, *as)
													isame[8] = reflect.DeepEqual(*xx, *xs)
													isame[10] = bls == beta
													if null {
														isame[11] = reflect.DeepEqual(*yy, *ys)
													} else {
														isame[11] = lzeresV(mat.General, mat.Full, 1, ml, ys, yy, abs(incy))
													}
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
													fatal = true
													goto label130
												}

												if !null {
													//                                Check the result.
													zmvch(trans, m, n, alpha, a, nmax, x, beta, y, yt, g, yy, eps, &err, &fatal, true, t)
													errmax = math.Max(errmax, err)
													//                                If got really bad answer, report and
													//                                return.
													if fatal {
														goto label130
													}
												} else {
													//                                Avoid repeating tests with M.le.0 or
													//                                N.le.0.
													goto label110
												}
											}
										}

									}
								}

							}

						}

					label110:
					}

				}
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 3460*2, t)
				} else if banded {
					passL2(sname, nc, 13828*2, t)
				}
			} else {
				t.Fail()
				fmt.Printf(" %6s completed the computational tests (%6d calls)\n ******* but with maximum test ratio%8.2f - suspect *******\n", sname, nc, errmax)
			}
			continue

		label130:
			;
			t.Fail()
			fmt.Printf(" ******* %6s FAILED ON CALL NUMBER:\n", sname)
			if full {
				fmt.Printf(" %6d: %6s('%c',%3d,%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d)         .\n", nc, sname, trans.Byte(), m, n, alpha, lda, incx, beta, incy)
			} else if banded {
				fmt.Printf(" %6d: %6s('%c',%3d,%3d,%3d,%3d,%4.1f, A,%3d, X,%2d,%4.1f, Y,%2d) .\n", nc, sname, trans.Byte(), m, n, kl, ku, alpha, lda, incx, beta, incy)
			}

		} else if sname == "Zhemv" || sname == "Zhbmv" || sname == "Zhpmv" {
			full := sname[2] == 'e'
			banded := sname[2] == 'b'
			packed := sname[2] == 'p'

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Hermitian
			optsfull.Style = mat.Hermitian
			if full {
				nargs = 10
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 11
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 9
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

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
						null = n <= 0

						for ic = 1; ic <= 2; ic++ {
							uplo = ichu[ic-1]
							opts.Uplo = uplo
							optsfull.Uplo = uplo

							//              Generate the matrix A.
							a = cmf(nmax, nmax, optsfull)
							aa = cmf(nmax, n, opts)
							zmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

							for ix = 1; ix <= ninc; ix++ {
								incx = inc[ix-1]
								lx = abs(incx) * n
								x.Inc, xx.Inc = incx, incx

								//                 Generate the vector X.
								x = cvf(n, incx)
								xx = cvf(lx, incx)
								zmakeL2V(1, n, x, 1, xx, abs(incx), 0, n-1, &reset, 0.5)
								if n > 1 {
									x.Set(n/2-1, zero)
									xx.Set(1+abs(incx)*(n/2-1)-1, zero)
								}

								for iy = 1; iy <= ninc; iy++ {
									incy = inc[iy-1]
									ly = abs(incy) * n
									y.Inc, yy.Inc = incy, incy

									for ia = 1; ia <= nalf; ia++ {
										alpha = alf.Get(ia - 1)

										for ib = 1; ib <= nbet; ib++ {
											beta = bet.Get(ib - 1)

											//                          Generate the vector Y.
											y = cvf(n, incy)
											yy = cvf(ly, incy)
											zmakeL2V(1, n, y, 1, yy, abs(incy), 0, n-1, &reset, 0.0)

											nc++

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
												err2 = Zhemv(uplo, n, alpha, aa, xx, beta, yy)
											} else if banded {
												err2 = Zhbmv(uplo, n, k, alpha, aa, xx, beta, yy)
											} else if packed {
												err2 = Zhpmv(uplo, n, alpha, aa.CVector(0, 0), xx, beta, yy)
											}

											//                          Check if error-exit was taken incorrectly.
											if !ok {
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
												isame[3] = reflect.DeepEqual(aa, as)
												isame[4] = ldas == lda
												isame[5] = reflect.DeepEqual(xx, xs)
												isame[6] = incxs == incx
												isame[7] = bls == beta
												if null {
													isame[8] = reflect.DeepEqual(yy, ys)
												} else {
													isame[8] = lzeresV(mat.General, Full, 1, n, ys, yy, abs(incy))
												}
												isame[9] = incys == incy
											} else if banded {
												isame[2] = ks == k
												isame[3] = als == alpha
												isame[4] = reflect.DeepEqual(aa, as)
												isame[5] = ldas == lda
												isame[6] = reflect.DeepEqual(xx, xs)
												isame[7] = incxs == incx
												isame[8] = bls == beta
												if null {
													isame[9] = reflect.DeepEqual(yy, ys)
												} else {
													isame[9] = lzeresV(mat.General, Full, 1, n, ys, yy, abs(incy))
												}
												isame[10] = incys == incy
											} else if packed {
												isame[2] = als == alpha
												isame[3] = reflect.DeepEqual(aa, as)
												isame[4] = reflect.DeepEqual(xx, xs)
												isame[5] = incxs == incx
												isame[6] = bls == beta
												if null {
													isame[7] = reflect.DeepEqual(yy, ys)
												} else {
													isame[7] = lzeresV(mat.General, Full, 1, n, ys, yy, abs(incy))
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
												zmvch(NoTrans, n, n, alpha, a, nmax, x, beta, y, yt, g, yy, eps, &err, &fatal, true, t)
												errmax = math.Max(errmax, err)
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
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 1441*2, t)
				} else if packed {
					passL2(sname, nc, 1441, t)
				} else if banded {
					passL2(sname, nc, 5761*2, t)
				}
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

		} else if sname == "Ztrmv" || sname == "Ztbmv" || sname == "Ztpmv" || sname == "Ztrsv" || sname == "Ztbsv" || sname == "Ztpsv" {
			z := cvf(nmax)
			full := sname[2] == 'r'
			banded := sname[2] == 'b'
			packed := sname[2] == 'p'

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Triangular
			optsfull.Style = mat.Triangular
			if full {
				nargs = 8
				opts.Storage = mat.Dense
			} else if banded {
				nargs = 9
				opts.Storage = mat.Banded
			} else if packed {
				nargs = 7
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			//     Set up zero vector for ZMVCH.
			for i = 1; i <= (nmax); i++ {
				z.Set(i-1, zero)
			}

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

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
						null = n <= 0

						for icu = 1; icu <= 2; icu++ {
							uplo = ichu[icu-1]
							opts.Uplo = uplo
							optsfull.Uplo = uplo

							for ict = 1; ict <= 3; ict++ {
								trans = icht[ict-1]

								for icd = 1; icd <= 2; icd++ {
									diag = ichd[icd-1]
									opts.Diag = diag
									optsfull.Diag = diag

									//                    Generate the matrix A.
									a = cmf(nmax, nmax, optsfull)
									aa = cmf(nmax, nmax, opts)
									zmakeL2M(n, n, a, nmax, aa, lda, k, k, &reset, 0.0)

									for ix = 1; ix <= ninc; ix++ {
										incx = inc[ix-1]
										lx = abs(incx) * n
										x.Inc, xx.Inc = incx, incx

										//                       Generate the vector X.
										x = cvf(n, incx)
										xx = cvf(lx, incx)
										zmakeL2V(1, n, x, 1, xx, abs(incx), 0, n-1, &reset, 0.5)
										if n > 1 {
											x.Set(n/2-1, zero)
											xx.Set(1+abs(incx)*(n/2-1)-1, zero)
										}

										nc++

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
										if string(sname[3:5]) == "mv" {
											if full {
												err2 = Ztrmv(uplo, trans, diag, n, aa, xx)
											} else if banded {
												err2 = Ztbmv(uplo, trans, diag, n, k, aa, xx)
											} else if packed {
												err2 = Ztpmv(uplo, trans, diag, n, aa.CVector(0, 0), xx)
											}
										} else if string(sname[3:5]) == "sv" {
											if full {
												err2 = Ztrsv(uplo, trans, diag, n, aa, xx)
											} else if banded {
												err2 = Ztbsv(uplo, trans, diag, n, k, aa, xx)
											} else if packed {
												err2 = Ztpsv(uplo, trans, diag, n, aa.CVector(0, 0), xx)
											}
										}

										//                       Check if error-exit was taken incorrectly.
										if !ok {
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
											isame[4] = reflect.DeepEqual(aa, as)
											isame[5] = ldas == lda
											if null {
												isame[6] = reflect.DeepEqual(xx, xs)
											} else {
												isame[6] = lzeresV(mat.General, Full, 1, n, xs, xx, abs(incx))
											}
											isame[7] = incxs == incx
										} else if banded {
											isame[4] = ks == k
											isame[5] = reflect.DeepEqual(aa, as)
											isame[6] = ldas == lda
											if null {
												isame[7] = reflect.DeepEqual(xx, xs)
											} else {
												isame[7] = lzeresV(mat.General, Full, 1, n, xs, xx, abs(incx))
											}
											isame[8] = incxs == incx
										} else if packed {
											isame[4] = reflect.DeepEqual(aa, as)
											if null {
												isame[5] = reflect.DeepEqual(xx, xs)
											} else {
												isame[5] = lzeresV(mat.General, Full, 1, n, xs, xx, abs(incx))
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
											z.Inc = incx
											if string(sname[3:5]) == "mv" {
												//                             Check the result.
												zmvch(trans, n, n, one, a, nmax, x, zero, z, xt, g, xx, eps, &err, &fatal, true, t)
											} else if string(sname[3:5]) == "sv" {
												//                             Compute approximation to original vector.
												for i = 1; i <= n; i++ {
													z.Set(i-1, xx.Get((i-1)*abs(incx)))
													xx.Set((i-1)*abs(incx), x.Get(i-1))
												}
												zmvch(trans, n, n, one, a, nmax, z, zero, x, xt, g, xx, eps, &err, &fatal, false, t)
											}
											errmax = math.Max(errmax, err)
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
			}

			//     Report result.
			if errmax < thresh {
				if full {
					passL2(sname, nc, 241*2, t)
				} else if packed {
					passL2(sname, nc, 241, t)
				} else if banded {
					passL2(sname, nc, 961*2, t)
				}
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

		} else if sname == "Zgerc" || sname == "Zgeru" {
			conj := sname[4] == 'c'

			opts.Style = mat.General
			opts.Storage = mat.Dense
			optsfull.Style = mat.General

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			for i := range isame {
				isame[i] = true
			}

			//     Define the number of arguments.
			nargs = 9

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				opts.Major = maj
				optsfull.Major = maj

				for in = 1; in <= nidim; in++ {
					n = idim[in-1]
					nd = n/2 + 1

					for im = 1; im <= 2; im++ {
						if im == 1 {
							m = max(n-nd, 0)
						}
						if im == 2 {
							m = min(n+nd, nmax)
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
						null = n <= 0 || m <= 0

						for ix = 1; ix <= ninc; ix++ {
							incx = inc[ix-1]
							lx = abs(incx) * m
							x.Inc, xx.Inc = incx, incx

							//              Generate the vector X.
							x = cvf(m, incx)
							xx = cvf(lx, incx)
							zmakeL2V(1, m, x, 1, xx, abs(incx), 0, m-1, &reset, 0.5)
							if m > 1 {
								x.Set(m/2-1, zero)
								xx.Set(1+abs(incx)*(m/2-1)-1, zero)
							}
							//
							for iy = 1; iy <= ninc; iy++ {
								incy = inc[iy-1]
								ly = abs(incy) * n
								y.Inc, yy.Inc = incy, incy

								//                 Generate the vector Y.
								y = cvf(n, incy)
								yy = cvf(ly, incy)
								zmakeL2V(1, n, y, 1, yy, abs(incy), 0, n-1, &reset, 0.0)
								if n > 1 {
									y.Set(n/2-1, zero)
									yy.Set(abs(incy)*(n/2-1), zero)
								}

								for ia = 1; ia <= nalf; ia++ {
									alpha = alf.Get(ia - 1)

									//                    Generate the matrix A.
									a = cmf(nmax, nmax, optsfull)
									aa = cmf(lda, n, opts)
									zmakeL2M(m, n, a, nmax, aa, lda, m-1, n-1, &reset, 0.0)

									nc = nc + 1

									//                    Save every datum before calling the subroutine.
									ms = m
									ns = n
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									xs = xx.DeepCopy()
									ys = yy.DeepCopy()

									//                    Call the subroutine.
									if conj {
										err2 = Zgerc(m, n, alpha, xx, yy, aa)
									} else {
										err2 = Zgeru(m, n, alpha, xx, yy, aa)
									}

									//                    Check if error-exit was taken incorrectly.
									if !ok {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										(fatal) = true
										goto label4140
									}

									//                    See what data changed inside subroutine.
									isame[0] = ms == m
									isame[1] = ns == n
									isame[2] = als == alpha
									isame[3] = reflect.DeepEqual(xx, xs)
									isame[5] = reflect.DeepEqual(yy, ys)
									if null {
										isame[7] = reflect.DeepEqual(aa, as)
									} else {
										isame[7] = lzeresM(m, n, as, aa, lda)
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
										goto label4140
									}

									if !null {
										a.ToColMajor()
										aa.ToColMajor()
										//                       Check the result column by column.
										z := cmf(m, nmax, a.Opts.DeepCopy())
										w := y.DeepCopy()
										if incx > 0 {
											for i = 1; i <= m; i++ {
												z.Set(i-1, 0, x.Get(i-1))
											}
										} else {
											for i = 1; i <= m; i++ {
												z.Set(i-1, 0, x.Get(m-i))
											}
										}
										for j = 1; j <= n; j++ {
											if incy > 0 {
												w.Set(0, y.Get(j-1))
											} else {
												w.Set(0, y.Get(n-j))
											}
											if conj {
												w.Set(0, w.GetConj(0))
											}
											zmvch(NoTrans, m, 1, alpha, z, nmax, w, one, a.CVector(0, j-1, 1), yt, g, aa.CVector(0, j-1), eps, &err, &fatal, true, t)
											errmax = math.Max(errmax, err)
											//                          If got really bad answer, report and return.
											if fatal {
												goto label4130
											}
										}
										a.ToRowMajor()
										aa.ToRowMajor()
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
			}

			//     Report result.
			if errmax < thresh {
				passL2(sname, nc, 388*2, t)
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

		} else if sname == "Zher" || sname == "Zhpr" {
			full := sname[2] == 'e'
			packed := sname[2] == 'p'

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 7
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 6
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

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

				for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
					if packed && maj == mat.Row {
						continue
					}
					opts.Major = maj
					optsfull.Major = maj

					for ic = 1; ic <= 2; ic++ {
						uplo = ichu[ic-1]
						upper = uplo == Upper
						opts.Uplo = uplo
						optsfull.Uplo = uplo

						for ix = 1; ix <= ninc; ix++ {
							incx = inc[ix-1]
							lx = abs(incx) * n
							x.Inc, xx.Inc = incx, incx

							//              Generate the vector X.
							x = cvf(n, incx)
							xx = cvf(lx, incx)
							zmakeL2V(1, n, x, 1, xx, abs(incx), 0, n-1, &reset, 0.5)
							if n > 1 {
								x.Set(n/2-1, zero)
								xx.Set(1+abs(incx)*(n/2-1)-1, zero)
							}

							for ia = 1; ia <= nalf; ia++ {
								ralpha := real(alf.Get(ia - 1))
								alpha = complex(ralpha, rzero)
								null = n <= 0 || ralpha == rzero

								//                 Generate the matrix A.
								a = cmf(nmax, nmax, optsfull)
								aa = cmf(lda, n, opts)
								// zmakeL2M(n, n, a, nmax, aa, lda, n-1, n-1, &reset, 0.0)

								nc++

								//                 Save every datum before calling the subroutine.
								uplos = uplo
								ns = n
								rals = ralpha
								as = aa.DeepCopy()
								ldas = lda
								xs = xx.DeepCopy()

								//                 Call the subroutine.
								if full {
									err2 = Zher(uplo, n, ralpha, xx, aa)
								} else if packed {
									err2 = Zhpr(uplo, n, ralpha, xx, aa.CVector(0, 0))
								}

								//                 Check if error-exit was taken incorrectly.
								if !ok {
									t.Fail()
									fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
									(fatal) = true
									goto label5120
								}

								//                 See what data changed inside subroutines.
								isame[0] = uplo == uplos
								isame[1] = ns == n
								isame[2] = rals == ralpha
								isame[3] = reflect.DeepEqual(xx, xs)
								if null {
									isame[5] = reflect.DeepEqual(aa, as)
								} else {
									isame[5] = lzeresM(n, n, as, aa, lda)
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
									a.ToColMajor()
									aa.ToColMajor()
									//                    Check the result column by column.
									z := cmf(n, 1, optsge)
									w := cvf(n)
									if incx > 0 {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(i-1))
										}
									} else {
										for i = 1; i <= n; i++ {
											z.Set(i-1, 0, x.Get(n-i))
										}
									}
									ja = 1
									for j = 1; j <= n; j++ {
										w.Set(0, z.GetConj(j-1, 0))
										if upper {
											jj = 1
											lj = j
										} else {
											jj = j
											lj = n - j + 1
										}
										zmvch(NoTrans, lj, 1, alpha, z.OffIdx(jj-1), lj, w, one, a.CVector(jj-1, j-1, 1), yt, g, aa.CVectorIdx(ja-1), eps, &err, &fatal, true, t)
										if full {
											if upper {
												ja += lda
											} else {
												ja += lda + 1
											}
										} else {
											ja += lj
										}
										errmax = math.Max(errmax, err)
										//                       If got really bad answer, report and return.
										if fatal {
											goto label5110
										}
									}
									a.ToRowMajor()
									aa.ToRowMajor()
								} else {
									//                    Avoid repeating tests if N.le.0.
									if n <= 0 {
										goto label5100
									}
								}

							}

						}

					}
				}

			label5100:
			}

			//     Report result.
			if errmax < thresh {
				if packed {
					passL2(sname, nc, 121, t)
				} else {
					passL2(sname, nc, 121*2-1, t)
				}
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

		} else if sname == "Zher2" || sname == "Zhpr2" {
			full := sname[2] == 'e'
			packed := sname[2] == 'p'

			if zchkeLevel2(sname); !common.infoc.ok {
				t.Fail()
			}

			//     Define the number of arguments.
			opts.Style = mat.Symmetric
			optsfull.Style = mat.Symmetric
			if full {
				nargs = 9
				opts.Storage = mat.Dense
			} else if packed {
				nargs = 8
				opts.Storage = mat.Packed
				opts.Major = mat.Col
				optsfull.Major = mat.Col
			}

			nc = 0

			for _, maj := range []mat.MatMajor{mat.Row, mat.Col} {
				if packed && maj == mat.Row {
					continue
				}
				opts.Major = maj
				optsfull.Major = maj

				for in = 1; in <= nidim; in++ {
					n = idim[in-1]
					//        Set LDA to 1 more than minimum value if room.
					lda = n
					if lda < (nmax) {
						lda++
					}
					//        Skip tests if not enough room.
					if lda > (nmax) {
						goto label6140
					}

					for ic = 1; ic <= 2; ic++ {
						uplo = ichu[ic-1]
						upper = uplo == Upper
						opts.Uplo = uplo
						optsfull.Uplo = uplo

						for ix = 1; ix <= ninc; ix++ {
							incx = inc[ix-1]
							lx = abs(incx) * n
							x.Inc, xx.Inc = incx, incx

							//              Generate the vector X.
							x = cvf(n, incx)
							xx = cvf(lx, incx)
							zmakeL2V(1, n, x, 1, xx, abs(incx), 0, n-1, &reset, 0.5)
							if n > 1 {
								x.Set(n/2-1, zero)
								xx.Set(1+abs(incx)*(n/2-1)-1, zero)
							}

							for iy = 1; iy <= ninc; iy++ {
								incy = inc[iy-1]
								ly = abs(incy) * n
								y.Inc, yy.Inc = incy, incy

								//                 Generate the vector Y.
								y = cvf(n, incy)
								yy = cvf(ly, incy)
								zmakeL2V(1, n, y, 1, yy, abs(incy), 0, n-1, &reset, 0.0)
								if n > 1 {
									y.Set(n/2-1, zero)
									yy.Set(1+abs(incy)*(n/2-1)-1, zero)
								}

								for ia = 1; ia <= nalf; ia++ {
									alpha = alf.Get(ia - 1)
									null = n <= 0 || alpha == zero

									//                    Generate the matrix A.
									a = cmf(nmax, nmax, optsfull)
									aa = cmf(lda, n, opts)
									// zmakeL2M(n, n, a, nmax, aa, lda, n-1, n-1, &reset, 0.0)

									nc++

									//                    Save every datum before calling the subroutine.
									uplos = uplo
									ns = n
									als = alpha
									as = aa.DeepCopy()
									ldas = lda
									xs = xx.DeepCopy()
									incxs = incx
									ys = yy.DeepCopy()
									incys = incy

									//                    Call the subroutine.
									if full {
										err2 = Zher2(uplo, n, alpha, xx, yy, aa)
									} else if packed {
										err2 = Zhpr2(uplo, n, alpha, xx, yy, aa.CVector(0, 0))
									}

									//                    Check if error-exit was taken incorrectly.
									if !ok {
										t.Fail()
										fmt.Printf(" ******* FATAL ERROR - ERROR-EXIT TAKEN ON VALID CALL *******\n")
										(fatal) = true
										goto label6160
									}

									//                    See what data changed inside subroutines.
									isame[0] = uplo == uplos
									isame[1] = ns == n
									isame[2] = als == alpha
									isame[3] = reflect.DeepEqual(xx, xs)
									isame[4] = incxs == incx
									isame[5] = reflect.DeepEqual(yy, ys)
									isame[6] = incys == incy
									if null {
										isame[7] = reflect.DeepEqual(aa, as)
									} else {
										isame[7] = lzeresM(n, n, as, aa, lda)
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
										a.ToColMajor()
										aa.ToColMajor()
										//                       Check the result column by column.
										z := cmf(n, 2, optsge)
										w := cvf(2)
										if incx > 0 {
											for i = 1; i <= n; i++ {
												z.Set(i-1, 0, x.Get(i-1))
											}
										} else {
											for i = 1; i <= n; i++ {
												z.Set(i-1, 0, x.Get(n-i))
											}
										}
										if incy > 0 {
											for i = 1; i <= n; i++ {
												z.Set(i-1, 1, y.Get(i-1))
											}
										} else {
											for i = 1; i <= n; i++ {
												z.Set(i-1, 1, y.Get(n-i))
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
											zmvch(NoTrans, lj, 2, one, z.OffIdx(jj-1), nmax, w, one, a.CVector(jj-1, j-1, 1), yt, g, aa.CVectorIdx(ja-1), eps, &err, &fatal, true, t)
											if full {
												if upper {
													ja += lda
												} else {
													ja += lda + 1
												}
											} else {
												ja += lj
											}
											errmax = math.Max(errmax, err)
											//                          If got really bad answer, report and return.
											if fatal {
												goto label6150
											}
										}
										a.ToRowMajor()
										aa.ToRowMajor()
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
			}

			//     Report result.
			if errmax < thresh {
				if packed {
					passL2(sname, nc, 481, t)
				} else {
					passL2(sname, nc, 481*2, t)
				}
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

func zmakeL2V(m, n int, a *mat.CVector, nmax int, aa *mat.CVector, lda, kl, ku int, reset *bool, transl complex128) {
	var rogue, zero complex128
	var i, j int

	zero = (0.0 + 0.0*1i)
	rogue = (-1.0e10 + 1.0e10*1i)

	//     Generate data in array A.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			if (i <= j && j-i <= ku) || (i >= j && i-j <= kl) {
				a.Set(i-1+(j-1)*nmax, zbeg(reset)+transl)
			} else {
				a.Set(i-1+(j-1)*nmax, zero)
			}
		}
	}

	//     Store elements in array AS in data structure required by routine.
	for j = 1; j <= n; j++ {
		for i = 1; i <= m; i++ {
			aa.Set(i-1+(j-1)*lda, a.Get(i-1+(j-1)*nmax))
		}
		for i = m + 1; i <= lda; i++ {
			aa.Set(i-1+(j-1)*lda, rogue)
		}
	}
}

func zmakeL2M(m, n int, a *mat.CMatrix, nmax int, aa *mat.CMatrix, lda, kl, ku int, reset *bool, transl complex128) {
	var one, rogue, zero complex128
	var rrogue float64
	var i, i1, i2, i3, ibeg, iend, ioff, j, kk int

	zero = (0.0 + 0.0*1i)
	one = (1.0 + 0.0*1i)
	rogue = (-1.0e10 + 1.0e10*1i)
	rrogue = -1.0e10

	gen := aa.Opts.Style == mat.General
	sym := aa.Opts.Style == mat.Symmetric
	tri := aa.Opts.Style == mat.Triangular
	upper := (sym || tri) && aa.Opts.Uplo == mat.Upper
	lower := (sym || tri) && aa.Opts.Uplo == mat.Lower
	unit := tri && aa.Opts.Diag == mat.Unit
	dense := aa.Opts.Storage == mat.Dense
	banded := aa.Opts.Storage == mat.Banded
	packed := aa.Opts.Storage == mat.Packed

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
	if gen && dense {
		for j = 1; j <= n; j++ {
			for i = 1; i <= m; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = m + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
		}
	} else if gen && banded {
		for j = 1; j <= n; j++ {
			for i1 = 1; i1 <= ku+1-j; i1++ {
				aa.Set(i1-1, j-1, rogue)
			}
			for i2 = i1; i2 <= min(kl+ku+1, ku+1+m-j); i2++ {
				aa.Set(i2-1, j-1, a.Get(i2+j-ku-1-1, j-1))
			}
			for i3 = i2; i3 <= lda; i3++ {
				aa.Set(i3-1, j-1, rogue)
			}
		}
	} else if (sym || tri) && dense {
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
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			if sym {
				aa.Set(j, j-1, complex(real(aa.Get(j, j-1)), rrogue))
			}
		}
	} else if (sym || tri) && banded {
		for j = 1; j <= n; j++ {
			if upper {
				kk = kl + 1
				ibeg = max(1, kl+2-j)
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
				iend = min(kl+1, 1+m-j)
			}
			for i = 1; i <= ibeg-1; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			for i = ibeg; i <= iend; i++ {
				aa.Set(i-1, j-1, a.Get(i+j-kk-1, j-1))
			}
			for i = iend + 1; i <= lda; i++ {
				aa.Set(i-1, j-1, rogue)
			}
			if sym {
				aa.Set(j, j-1, complex(real(aa.Get(j, j-1)), rrogue))
			}
		}
	} else if (sym || tri) && packed {
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
				aa.Set(ioff-1, 0, a.Get(i-1, j-1))
				if i == j {
					if unit {
						aa.Set(ioff-1, 0, rogue)
					}
					if sym {
						aa.Set(ioff-1, 0, complex(real(aa.Get(ioff-1, 0)), rrogue))
					}
				}
			}
		}
	}
}

func zmvch(trans mat.MatTrans, m, n int, alpha complex128, a *mat.CMatrix, nmax int, x *mat.CVector, beta complex128, y, yt *mat.CVector, g *mat.Vector, yy *mat.CVector, eps float64, err *float64, fatal *bool, mv bool, t *testing.T) {
	var ctran, tran bool
	var zero complex128
	var erri, rone, rzero float64
	var i, incxl, incyl, iy, j, jx, kx, ky, ml, nl int

	zero = (0.0 + 0.0*1i)
	rzero = 0.0
	rone = 1.0

	tran = trans == Trans
	ctran = trans == ConjTrans
	if tran || ctran {
		ml = n
		nl = m
	} else {
		ml = m
		nl = n
	}
	if x.Inc < 0 {
		kx = nl
		incxl = -1
	} else {
		kx = 1
		incxl = 1
	}
	if y.Inc < 0 {
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
				g.Set(iy-1, g.Get(iy-1)+abs1(a.Get(j-1, i-1))*abs1(x.Get(jx-1)))
				jx += incxl
			}
		} else if ctran {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.GetConj(j-1, i-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+abs1(a.Get(j-1, i-1))*abs1(x.Get(jx-1)))
				jx += incxl
			}
		} else {
			for j = 1; j <= nl; j++ {
				yt.Set(iy-1, yt.Get(iy-1)+a.Get(i-1, j-1)*x.Get(jx-1))
				g.Set(iy-1, g.Get(iy-1)+abs1(a.Get(i-1, j-1))*abs1(x.Get(jx-1)))
				jx += incxl
			}
		}
		yt.Set(iy-1, alpha*yt.Get(iy-1)+beta*y.Get(iy-1))
		g.Set(iy-1, abs1(alpha)*g.Get(iy-1)+abs1(beta)*abs1(y.Get(iy-1)))
		iy += incyl
	}

	//     Compute the error ratio for this result.
	(*err) = rzero
	for i = 1; i <= ml; i++ {
		erri = cmplx.Abs(yt.Get(i-1)-yy.Get((i-1)*abs(y.Inc))) / eps
		if g.Get(i-1) != rzero {
			erri /= g.Get(i - 1)
		}
		(*err) = math.Max(*err, erri)
		if (*err)*math.Sqrt(eps) >= rone {
			goto label60
		}
	}
	//     If the loop completes, all results are at least half accurate.
	return

	//     Report fatal error.
label60:
	;
	*fatal = true
	fmt.Printf(" ******* FATAL ERROR - COMPUTED RESULT IS LESS THAN HALF ACCURATE *******\n                       EXPECTED RESULT                    COMPUTED RESULT\n")
	for i = 0; i < ml; i++ {
		if mv {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, yt.Get(i), yy.Get(i))
		} else {
			fmt.Printf(" %7d  (%15.6f,%15.6f)\n", i, yy.Get(i), yt.Get(i))
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
	if string(_type) == "GE" || string(_type) == "ge" {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					goto label70
				}
			}
		}
	} else if string(_type) == "HE" || string(_type) == "he" {
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

func lzeresV(_type mat.MatStyle, uplo mat.MatUplo, m, n int, aa, as *mat.CVector, lda int) bool {
	var upper bool
	var i, ibeg, iend, j int

	upper = uplo == Upper
	if _type == mat.General {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= lda; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
		}
	} else if _type == mat.Symmetric || _type == mat.Hermitian {
		for j = 1; j <= n; j++ {
			if upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
			for i = iend + 1; i <= lda; i++ {
				if aa.Get(i-1+(j-1)*lda) != as.Get(i-1+(j-1)*lda) {
					return false
				}
			}
		}
	}

	return true
}

func lzeresM(m, n int, aa, as *mat.CMatrix, lda int) bool {
	var i, ibeg, iend, j int

	if aa.Opts.Style == mat.General && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			for i = m + 1; i <= aa.Rows; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	} else if (aa.Opts.Style == mat.Symmetric || aa.Opts.Style == mat.Hermitian) && aa.Opts.Storage == mat.Dense {
		for j = 1; j <= n; j++ {
			if aa.Opts.Uplo == mat.Upper {
				ibeg = 1
				iend = j
			} else {
				ibeg = j
				iend = n
			}
			for i = 1; i <= ibeg-1; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
			for i = iend + 1; i <= aa.Rows; i++ {
				if aa.Get(i-1, j-1) != as.Get(i-1, j-1) {
					return false
				}
			}
		}
	}

	return true
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

func zchkeLevel2(srnamt string) {
	var err error

	x := cvf(1)
	y := cvf(1)
	a := cmf(1, 1, opts)

	//
	//  Tests the error exits from the Level 2 Blas.
	//  Requires a special version of the error-handling routine XERBLA.
	//  ALPHA, RALPHA, BETA, A, X and Y should not need to be defined.
	//
	//  Auxiliary routine for test program for Level 2 Blas.
	//
	//  -- Written on 10-August-1987.
	//     Richard Hanson, Sandia National Labs.
	//     Jeremy Du Croz, NAG Central Office.
	//
	one := 1.0
	two := 2.0

	errt := &common.infoc.errt
	ok := &common.infoc.ok
	lerr := &common.infoc.lerr

	*ok = true
	*lerr = true

	alpha := complex(one, -one)
	beta := complex(two, -two)
	ralpha := one

	switch srnamt {
	case "Zgemv":
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Zgemv('/', 0, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgemv(NoTrans, -1, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgemv(NoTrans, 0, -1, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zgemv(NoTrans, 2, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
	case "Zgbmv":
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Zgbmv('/', 0, 0, 0, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgbmv(NoTrans, -1, 0, 0, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgbmv(NoTrans, 0, -1, 0, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("kl invalid: -1")
		err = Zgbmv(NoTrans, 0, 0, -1, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("ku invalid: -1")
		err = Zgbmv(NoTrans, 2, 0, 0, -1, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zgbmv(NoTrans, 0, 0, 1, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
	case "Zhemv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zhemv('/', 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhemv(Upper, -1, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zhemv(Upper, 2, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
	case "Zhbmv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zhbmv('/', 0, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhbmv(Upper, -1, 0, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Zhbmv(Upper, 0, -1, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zhbmv(Upper, 0, 1, alpha, a, x, beta, y)
		Chkxer(srnamt, err)
	case "Zhpmv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zhpmv('/', 0, alpha, a.CVectorIdx(0), x, beta, y)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhpmv(Upper, -1, alpha, a.CVectorIdx(0), x, beta, y)
		Chkxer(srnamt, err)
	case "Ztrmv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztrmv('/', NoTrans, NonUnit, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztrmv(Upper, '/', NonUnit, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztrmv(Upper, NoTrans, '/', 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrmv(Upper, NoTrans, NonUnit, -1, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Ztrmv(Upper, NoTrans, NonUnit, 2, a, x)
		Chkxer(srnamt, err)
	case "Ztbmv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztbmv('/', NoTrans, NonUnit, 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztbmv(Upper, '/', NonUnit, 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztbmv(Upper, NoTrans, '/', 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztbmv(Upper, NoTrans, NonUnit, -1, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Ztbmv(Upper, NoTrans, NonUnit, 0, -1, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Ztbmv(Upper, NoTrans, NonUnit, 0, 1, a, x)
		Chkxer(srnamt, err)
	case "Ztpmv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztpmv('/', NoTrans, NonUnit, 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztpmv(Upper, '/', NonUnit, 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztpmv(Upper, NoTrans, '/', 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztpmv(Upper, NoTrans, NonUnit, -1, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
	case "Ztrsv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztrsv('/', NoTrans, NonUnit, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztrsv(Upper, '/', NonUnit, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztrsv(Upper, NoTrans, '/', 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztrsv(Upper, NoTrans, NonUnit, -1, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Ztrsv(Upper, NoTrans, NonUnit, 2, a, x)
		Chkxer(srnamt, err)
	case "Ztbsv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztbsv('/', NoTrans, NonUnit, 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztbsv(Upper, '/', NonUnit, 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztbsv(Upper, NoTrans, '/', 0, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztbsv(Upper, NoTrans, NonUnit, -1, 0, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("k invalid: -1")
		err = Ztbsv(Upper, NoTrans, NonUnit, 0, -1, a, x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Ztbsv(Upper, NoTrans, NonUnit, 0, 1, a, x)
		Chkxer(srnamt, err)
	case "Ztpsv":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Ztpsv('/', NoTrans, NonUnit, 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("trans invalid: Unrecognized: /")
		err = Ztpsv(Upper, '/', NonUnit, 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("diag invalid: Unrecognized: /")
		err = Ztpsv(Upper, NoTrans, '/', 0, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Ztpsv(Upper, NoTrans, NonUnit, -1, a.CVectorIdx(0), x)
		Chkxer(srnamt, err)
	case "Zgerc":
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgerc(-1, 0, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgerc(0, -1, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zgerc(2, 0, alpha, x, y, a)
		Chkxer(srnamt, err)
	case "Zgeru":
		*errt = fmt.Errorf("m invalid: -1")
		err = Zgeru(-1, 0, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zgeru(0, -1, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zgeru(2, 0, alpha, x, y, a)
		Chkxer(srnamt, err)
	case "Zher":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zher('/', 0, ralpha, x, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher(Upper, -1, ralpha, x, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zher(Upper, 2, ralpha, x, a)
		Chkxer(srnamt, err)
	case "Zhpr":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zhpr('/', 0, ralpha, x, a.CVectorIdx(0))
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhpr(Upper, -1, ralpha, x, a.CVectorIdx(0))
		Chkxer(srnamt, err)
	case "Zher2":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zher2('/', 0, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zher2(Upper, -1, alpha, x, y, a)
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("a.Rows invalid: 1 < 2")
		err = Zher2(Upper, 2, alpha, x, y, a)
		Chkxer(srnamt, err)
	case "Zhpr2":
		*errt = fmt.Errorf("uplo invalid: Unrecognized: /")
		err = Zhpr2('/', 0, alpha, x, y, a.CVectorIdx(0))
		Chkxer(srnamt, err)
		*errt = fmt.Errorf("n invalid: -1")
		err = Zhpr2(Upper, -1, alpha, x, y, a.CVectorIdx(0))
		Chkxer(srnamt, err)
	}

	if *ok {
		fmt.Printf(" %6s passed the tests of error-exits\n", srnamt)
	} else {
		fmt.Printf(" ******* %6s FAILED THE TESTS OF ERROR-EXITS *******\n", srnamt)
	}
}
